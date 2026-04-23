import asyncio, json, logging, os, re, traceback
from datetime import datetime
from pathlib import Path

import ollama
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE_DIR  = Path(__file__).parent
LOGS_DIR  = BASE_DIR / "logs"
MEM_FILE  = BASE_DIR / "memory" / "user_memory.json"
LOGS_DIR.mkdir(exist_ok=True)

# ── Loggers ──────────────────────────────────────────────────────────────────
def _make_logger(name, file, level=logging.DEBUG):
    log = logging.getLogger(name)
    log.setLevel(level)
    fh = logging.FileHandler(LOGS_DIR / file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    if not log.handlers:
        log.addHandler(fh)
    return log

agent_log = _make_logger("agent",  "agent.log")
error_log = _make_logger("errors", "errors.log", logging.ERROR)

# ── Memory ───────────────────────────────────────────────────────────────────
def load_memory() -> dict:
    if MEM_FILE.exists():
        try:
            return json.loads(MEM_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_memory(data: dict):
    MEM_FILE.parent.mkdir(exist_ok=True)
    MEM_FILE.write_text(json.dumps(data, indent=2))

def inject_memory(value: str, memory: dict) -> str:
    """Replace {{key}} placeholders with memory values."""
    for k, v in memory.items():
        value = value.replace(f"{{{{{k}}}}}", str(v))
    return value

# ── Page info ────────────────────────────────────────────────────────────────
async def get_page_info(page) -> dict:
    try:
        url   = page.url
        title = await page.title()
        text  = (await page.inner_text("body"))[:3000]
        elems = await page.evaluate("""() => {
            const sel = 'a,button,input,select,textarea,[role=button],[role=link]';
            return [...document.querySelectorAll(sel)].slice(0,60).map(el => ({
                tag:         el.tagName.toLowerCase(),
                type:        el.type         || '',
                id:          el.id           || '',
                name:        el.name         || '',
                placeholder: el.placeholder  || '',
                text:        (el.innerText || el.value || el.placeholder || '').trim().slice(0,80),
                href:        el.href         || '',
                selector:    el.id ? '#'+el.id : (el.name ? '[name="'+el.name+'"]' : '')
            }));
        }""")
        return {"url": url, "title": title, "text": text, "elements": elems}
    except Exception as e:
        return {"url": page.url, "title": "", "text": "", "elements": [], "error": str(e)}

# ── Robust fill helper ────────────────────────────────────────────────────────
async def robust_fill(page, selector: str, value: str, timeout: int = 30000):
    """
    Fill a field reliably without using .clear() which LinkedIn and similar
    sites block. Tries multiple strategies in order:
    1. page.fill() directly (no pre-clear needed)
    2. Click → Ctrl+A → Delete → keyboard.type()
    3. JS value injection + input/change events (React-compatible)
    """
    try:
        await page.wait_for_selector(selector, state="visible", timeout=timeout)
        await page.fill(selector, value, timeout=timeout)
        return
    except Exception as e1:
        agent_log.warning(f"fill() failed for {selector}: {e1} — trying click+type")

    try:
        loc = page.locator(selector).first
        await loc.wait_for(state="visible", timeout=timeout)
        await loc.click(timeout=10000)
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Delete")
        await page.keyboard.type(value, delay=50)
        return
    except Exception as e2:
        agent_log.warning(f"click+type failed for {selector}: {e2} — trying JS inject")

    # Last resort: inject value via JS and fire React-compatible events
    await page.evaluate(f"""
        (function() {{
            const el = document.querySelector('{selector}');
            if (!el) return;
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(el, {json.dumps(value)});
            el.dispatchEvent(new Event('input',  {{bubbles:true}}));
            el.dispatchEvent(new Event('change', {{bubbles:true}}));
        }})();
    """)

async def robust_fill_by_label(page, label: str, value: str, timeout: int = 30000):
    """Fill by visible label text when no reliable selector is available."""
    try:
        loc = page.get_by_label(label, exact=False).first
        await loc.wait_for(state="visible", timeout=timeout)
        await loc.fill(value, timeout=timeout)
        return
    except Exception:
        pass
    # fallback: by placeholder
    try:
        loc = page.get_by_placeholder(label, exact=False).first
        await loc.wait_for(state="visible", timeout=timeout)
        await loc.fill(value, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Could not fill field labeled '{label}': {e}")

# ── LLM call ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a browser automation agent. Given:
- A user task
- The user's memory (credentials, preferences, etc.)
- The current browser page state (URL, title, visible text, interactive elements)
- The history of previous actions taken

Return ONLY a valid JSON array of action objects. No explanation, no markdown fences.

Available actions:
  {"action":"navigate",         "args":{"url":"https://..."},                          "reason":"..."}
  {"action":"click",            "args":{"text":"Sign In"},                              "reason":"..."}
  {"action":"click_sel",        "args":{"selector":"#login-btn"},                       "reason":"..."}
  {"action":"fill",             "args":{"selector":"#email","value":"{{key}}"},          "reason":"..."}
  {"action":"fill_label",       "args":{"label":"Email","value":"{{key}}"},              "reason":"..."}
  {"action":"fill_placeholder", "args":{"placeholder":"Email","value":"{{key}}"},        "reason":"..."}
  {"action":"press",            "args":{"key":"Enter"},                                  "reason":"..."}
  {"action":"wait",             "args":{"ms":2000},                                     "reason":"..."}
  {"action":"wait_selector",    "args":{"selector":"#someEl","timeout":15000},           "reason":"..."}
  {"action":"scroll",           "args":{"direction":"down","amount":500},                "reason":"..."}
  {"action":"screenshot",       "args":{},                                               "reason":"..."}
  {"action":"done",             "args":{"message":"Task completed: ..."},                "reason":"..."}
  {"action":"error",            "args":{"message":"Cannot complete because: ..."},       "reason":"..."}

Rules:
- Use {{key}} placeholders for credentials — the engine replaces them before execution
- Prefer fill_label or fill_placeholder over fill when selector is uncertain
- After submitting a form always wait 2000ms before the next action
- Return 1-5 actions per response; you will be called again with updated page state
- If the task is clearly done, return a done action
- If something is impossible, return an error action
"""

def call_llm(model: str, task: str, memory: dict, page_info: dict, history: list) -> list:
    safe_mem = {
        k: ("***" if any(s in k.lower() for s in ["password","pass","pin","secret","card","token","key","ssn","cvv"]) else v)
        for k, v in memory.items()
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": json.dumps({
            "task":    task,
            "memory":  safe_mem,
            "page":    page_info,
            "history": history[-8:]
        }, indent=2)}
    ]
    resp = ollama.chat(model=model, messages=messages)
    raw  = resp["message"]["content"].strip()
    raw  = re.sub(r"^```(?:json)?\s*", "", raw)
    raw  = re.sub(r"\s*```$",          "", raw)
    actions = json.loads(raw)
    if not isinstance(actions, list):
        raise ValueError("LLM did not return a JSON array")
    return actions

# ── Action executor ───────────────────────────────────────────────────────────
# Generous timeouts for slow-loading sites like LinkedIn
NAV_TIMEOUT     = 30_000   # page.goto
VISIBLE_TIMEOUT = 30_000   # wait_for visible
CLICK_TIMEOUT   = 15_000
FILL_TIMEOUT    = 30_000

async def execute_action(page, action: dict, memory: dict, emit_fn=None) -> str:
    act    = action.get("action", "")
    args   = action.get("args", {})
    reason = action.get("reason", "")

    def log(msg):
        agent_log.info(msg)
        if emit_fn:
            emit_fn("agent_log", {"msg": msg})

    log(f"ACTION: {act} | args={args} | reason={reason}")

    try:
        # ── navigate ──────────────────────────────────────────────────────────
        if act == "navigate":
            url = args["url"]
            await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await page.wait_for_timeout(1500)
            return f"Navigated to {url}"

        # ── click by visible text ─────────────────────────────────────────────
        elif act == "click":
            txt = args["text"]
            loc = page.get_by_text(txt, exact=False).first
            await loc.wait_for(state="visible", timeout=VISIBLE_TIMEOUT)
            await loc.click(timeout=CLICK_TIMEOUT)
            await page.wait_for_timeout(1000)
            return f"Clicked text: {txt}"

        # ── click by CSS selector ─────────────────────────────────────────────
        elif act == "click_sel":
            sel = args["selector"]
            await page.wait_for_selector(sel, state="visible", timeout=VISIBLE_TIMEOUT)
            await page.click(sel, timeout=CLICK_TIMEOUT)
            await page.wait_for_timeout(1000)
            return f"Clicked selector: {sel}"

        # ── fill by CSS selector (robust — no .clear()) ───────────────────────
        elif act == "fill":
            sel = args["selector"]
            val = inject_memory(args.get("value", ""), memory)
            await robust_fill(page, sel, val, timeout=FILL_TIMEOUT)
            return f"Filled {sel}"

        # ── fill by ARIA label ────────────────────────────────────────────────
        elif act == "fill_label":
            label = args["label"]
            val   = inject_memory(args.get("value", ""), memory)
            await robust_fill_by_label(page, label, val, timeout=FILL_TIMEOUT)
            return f"Filled field labeled '{label}'"

        # ── fill by placeholder ───────────────────────────────────────────────
        elif act == "fill_placeholder":
            ph  = args["placeholder"]
            val = inject_memory(args.get("value", ""), memory)
            loc = page.get_by_placeholder(ph, exact=False).first
            await loc.wait_for(state="visible", timeout=FILL_TIMEOUT)
            await loc.fill(val, timeout=FILL_TIMEOUT)
            return f"Filled placeholder '{ph}'"

        # ── keyboard press ────────────────────────────────────────────────────
        elif act == "press":
            key = args.get("key", "Enter")
            await page.keyboard.press(key)
            await page.wait_for_timeout(1000)
            return f"Pressed {key}"

        # ── wait ms ───────────────────────────────────────────────────────────
        elif act == "wait":
            ms = int(args.get("ms", 2000))
            await page.wait_for_timeout(ms)
            return f"Waited {ms}ms"

        # ── wait for selector ─────────────────────────────────────────────────
        elif act == "wait_selector":
            sel = args["selector"]
            t   = int(args.get("timeout", 15000))
            await page.wait_for_selector(sel, state="visible", timeout=t)
            return f"Selector appeared: {sel}"

        # ── scroll ────────────────────────────────────────────────────────────
        elif act == "scroll":
            direction = args.get("direction", "down")
            amount    = int(args.get("amount", 500))
            dy = amount if direction == "down" else -amount
            await page.mouse.wheel(0, dy)
            await page.wait_for_timeout(500)
            return f"Scrolled {direction} {amount}px"

        # ── screenshot ────────────────────────────────────────────────────────
        elif act == "screenshot":
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(LOGS_DIR / f"screenshot_{ts}.png")
            await page.screenshot(path=path, full_page=False)
            log(f"Screenshot saved: {path}")
            if emit_fn:
                emit_fn("screenshot", {"path": os.path.basename(path)})
            return f"Screenshot: {path}"

        # ── done ──────────────────────────────────────────────────────────────
        elif act == "done":
            return "DONE: " + args.get("message", "Task complete")

        # ── error ─────────────────────────────────────────────────────────────
        elif act == "error":
            msg = "ERROR: " + args.get("message", "Unknown error")
            error_log.error(msg)
            return msg

        else:
            msg = f"UNKNOWN action: {act}"
            error_log.error(msg)
            return msg

    except PWTimeout as e:
        msg = f"Timeout in action '{act}': {e}"
        error_log.error(msg)
        return f"TIMEOUT: {msg}"
    except Exception as e:
        msg = f"Error in action '{act}': {e}\n{traceback.format_exc()}"
        error_log.error(msg)
        return f"FAIL: {msg}"

# ── Main task runner ──────────────────────────────────────────────────────────
async def run_task(task: str, model: str = "llama3.2", headless: bool = False, emit_fn=None) -> dict:
    memory  = load_memory()
    history = []
    final   = {"status": "unknown", "message": "", "screenshots": []}

    def log(msg):
        agent_log.info(msg)
        if emit_fn:
            emit_fn("agent_log", {"msg": msg})

    log(f"=== NEW TASK: {task} | model={model} ===")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless, args=["--no-sandbox"])
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        await page.goto("about:blank")

        for round_n in range(1, 13):
            log(f"--- Round {round_n}/12 ---")
            page_info = await get_page_info(page)
            log(f"Page: {page_info['url']} | {page_info['title']}")

            try:
                actions = call_llm(model, task, memory, page_info, history)
            except Exception as e:
                msg = f"LLM error: {e}"
                error_log.error(msg)
                log(f"LLM ERROR: {e}")
                final = {"status": "error", "message": msg}
                break

            done = False
            for action in actions:
                result = await execute_action(page, action, memory, emit_fn)
                history.append({"action": action, "result": result})
                log(f"Result: {result}")

                if result.startswith("DONE:"):
                    final = {"status": "success", "message": result[5:].strip()}
                    done = True; break
                if result.startswith("ERROR:") or result.startswith("FAIL:"):
                    final = {"status": "error", "message": result}
                    done = True; break

            if done:
                break
        else:
            final = {"status": "timeout", "message": "Max rounds reached without completion"}
            error_log.error(final["message"])

        # final screenshot
        try:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            spath = str(LOGS_DIR / f"final_{ts}.png")
            await page.screenshot(path=spath)
            final["screenshots"].append(os.path.basename(spath))
            if emit_fn:
                emit_fn("screenshot", {"path": os.path.basename(spath)})
        except Exception:
            pass

        await browser.close()

    log(f"=== TASK DONE: {final['status']} | {final['message']} ===")
    return final
