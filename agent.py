import asyncio, base64, json, logging, os, re, traceback
from datetime import datetime
from pathlib import Path
from collections import deque

import ollama
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE_DIR  = Path(__file__).parent
LOGS_DIR  = BASE_DIR / "logs"
MEM_FILE  = BASE_DIR / "memory" / "user_memory.json"
DOM_JS    = BASE_DIR / "dom_extract.js"
LOGS_DIR.mkdir(exist_ok=True)

GPU_CFG = {
    "num_gpu":    -1,
    "num_thread": os.cpu_count() or 4,
}

# ── Loggers ───────────────────────────────────────────────────────────────────
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

# ── Memory ────────────────────────────────────────────────────────────────────
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
    for k, v in memory.items():
        value = value.replace(f"{{{{{k}}}}}", str(v))
    return value

# ── Warmup ────────────────────────────────────────────────────────────────────
def warmup_model(model: str) -> bool:
    try:
        agent_log.info(f"Warming up model: {model}")
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            options={**GPU_CFG, "num_predict": 1},
        )
        agent_log.info(f"Model {model} warmed up.")
        return True
    except Exception as e:
        error_log.error(f"warmup_model failed for '{model}': {e}")
        return False

# ── PHASE 1: Indexed DOM extraction ──────────────────────────────────────────
_DOM_JS_SRC = None

def _load_dom_js() -> str:
    global _DOM_JS_SRC
    if _DOM_JS_SRC is None:
        if DOM_JS.exists():
            _DOM_JS_SRC = DOM_JS.read_text()
        else:
            _DOM_JS_SRC = """() => {
  const INTERACTIVE = 'a,button,input,select,textarea,[role=button],[role=link],[role=checkbox],[role=menuitem],[role=tab]';
  const map = {}; const lines = []; let idx = 0;
  const isVisible = el => {
    const r = el.getBoundingClientRect();
    if (r.width===0||r.height===0) return false;
    const s = window.getComputedStyle(el);
    return s.display!=='none'&&s.visibility!=='hidden'&&s.opacity!=='0';
  };
  document.querySelectorAll(INTERACTIVE).forEach(el => {
    if (!isVisible(el)||idx>=150) return;
    const tag=el.tagName.toLowerCase();
    const ph=el.placeholder?` placeholder="${el.placeholder.slice(0,40)}"`:''
    const href=el.href?` href="${el.href.slice(0,80)}"`:''
    const aria=el.getAttribute('aria-label')?` aria-label="${el.getAttribute('aria-label').slice(0,60)}"`:''
    const sel=el.id?'#'+el.id:el.name?'[name="'+el.name+'"]':'';
    const text=(el.getAttribute('aria-label')||el.placeholder||el.innerText?.trim()||el.value||'').slice(0,80).replace(/\\s+/g,' ');
    map[idx]={el,selector:sel,tag};
    lines.push(`[${idx}]<${tag}${ph}${href}${aria}>${text}</${tag}>`);
    idx++;
  });
  window.__AGENT_ELEM_MAP=map;
  return {elements:lines,count:idx};
}"""
    return _DOM_JS_SRC

async def get_page_info(page, capture_screenshot=False) -> dict:
    """Phase 1: Indexed DOM. Phase 2: optional screenshot for vision models."""
    try:
        url   = page.url
        title = await page.title()
        text  = (await page.inner_text("body"))[:2000]

        dom_result = await page.evaluate(_load_dom_js())
        elements   = dom_result.get("elements", [])

        info = {
            "url":        url,
            "title":      title,
            "text":       text,
            "elements":   elements,
            "elem_count": dom_result.get("count", 0),
        }

        if capture_screenshot:
            try:
                img_bytes = await page.screenshot(type="jpeg", quality=60,
                                                   full_page=False,
                                                   clip={"x":0,"y":0,"width":1280,"height":800})
                info["screenshot_b64"] = base64.b64encode(img_bytes).decode()
            except Exception as e:
                agent_log.warning(f"Screenshot capture failed: {e}")

        return info
    except Exception as e:
        return {"url": page.url, "title": "", "text": "", "elements": [], "error": str(e)}

# ── Phase 1: Click / fill by index ───────────────────────────────────────────
async def click_by_index(page, index: int) -> str:
    try:
        result = await page.evaluate("""(idx) => {
            const entry = window.__AGENT_ELEM_MAP?.[idx];
            if (!entry) return 'NOT_FOUND';
            entry.el.click();
            return entry.selector || entry.tag;
        }""", index)
        if result == "NOT_FOUND":
            raise ValueError(f"Element index {index} not found in map")
        await page.wait_for_timeout(1000)
        return f"Clicked element [{index}] ({result})"
    except Exception as e:
        raise RuntimeError(f"click_by_index({index}) failed: {e}")

async def fill_by_index(page, index: int, value: str) -> str:
    try:
        result = await page.evaluate("""(args) => {
            const entry = window.__AGENT_ELEM_MAP?.[args.idx];
            if (!entry) return 'NOT_FOUND';
            const el = entry.el;
            el.focus();
            const niv = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value');
            if (niv) niv.set.call(el, args.value);
            else el.value = args.value;
            el.dispatchEvent(new Event('input',  {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
            return entry.selector || entry.tag;
        }""", {"idx": index, "value": value})
        if result == "NOT_FOUND":
            raise ValueError(f"Element index {index} not found in map")
        return f"Filled element [{index}] with value"
    except Exception as e:
        raise RuntimeError(f"fill_by_index({index}) failed: {e}")

# ── Robust fill (fallback) ────────────────────────────────────────────────────
async def robust_fill(page, selector: str, value: str, timeout: int = 30000):
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
    await page.evaluate(f"""(function(){{
        const el=document.querySelector('{selector}');
        if(!el)return;
        const niv=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value');
        if(niv)niv.set.call(el,{json.dumps(value)});
        else el.value={json.dumps(value)};
        el.dispatchEvent(new Event('input',{{bubbles:true}}));
        el.dispatchEvent(new Event('change',{{bubbles:true}}));
    }})();""")

async def robust_fill_by_label(page, label: str, value: str, timeout: int = 30000):
    try:
        loc = page.get_by_label(label, exact=False).first
        await loc.wait_for(state="visible", timeout=timeout)
        await loc.fill(value, timeout=timeout)
        return
    except Exception:
        pass
    try:
        loc = page.get_by_placeholder(label, exact=False).first
        await loc.wait_for(state="visible", timeout=timeout)
        await loc.fill(value, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Could not fill field labeled '{label}': {e}")

# ── Phase 3: History summarization ───────────────────────────────────────────
def summarize_history(history: list, model: str, window: int = 6) -> list:
    """
    Keep last `window` actions verbatim.
    Summarize older entries into a single compressed entry.
    Prevents prompt bloat on long multi-step tasks.
    """
    if len(history) <= window:
        return history

    old    = history[:-window]
    recent = history[-window:]

    summary_text = "; ".join(
        f"{h['action'].get('action','?')}\u2192{str(h['result'])[:60]}"
        for h in old
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Summarize these browser automation steps in \u22642 sentences:\n{summary_text}"
            }],
            options={**GPU_CFG, "num_predict": 120},
        )
        summary = resp["message"]["content"].strip()
    except Exception:
        summary = f"Previously completed {len(old)} steps: {summary_text[:200]}"

    return [{"action": {"action": "summary"}, "result": summary}] + recent

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a browser automation agent. Given:
- A user task
- The user's memory (credentials, preferences, etc.)
- The current browser page state (URL, title, visible text, indexed interactive elements)
- The history of previous actions taken

Return ONLY a valid JSON array of action objects. No explanation, no markdown fences.

PREFERRED actions (index-based — more reliable):
  {"action":"click_index",      "args":{"index":3},                                  "reason":"..."}
  {"action":"fill_index",       "args":{"index":5, "value":"{{key}}"},               "reason":"..."}

FALLBACK actions (use when index not available):
  {"action":"navigate",         "args":{"url":"https://..."},                        "reason":"..."}
  {"action":"click",            "args":{"text":"Sign In"},                           "reason":"..."}
  {"action":"click_sel",        "args":{"selector":"#login-btn"},                    "reason":"..."}
  {"action":"fill",             "args":{"selector":"#email","value":"{{key}}"},      "reason":"..."}
  {"action":"fill_label",       "args":{"label":"Email","value":"{{key}}"},          "reason":"..."}
  {"action":"fill_placeholder", "args":{"placeholder":"Email","value":"{{key}}"},    "reason":"..."}
  {"action":"press",            "args":{"key":"Enter"},                              "reason":"..."}
  {"action":"wait",             "args":{"ms":2000},                                  "reason":"..."}
  {"action":"wait_selector",    "args":{"selector":"#someEl","timeout":15000},       "reason":"..."}
  {"action":"scroll",           "args":{"direction":"down","amount":500},            "reason":"..."}
  {"action":"select_option",    "args":{"selector":"#qty","value":"2"},              "reason":"..."}
  {"action":"screenshot",       "args":{},                                            "reason":"..."}
  {"action":"new_tab",          "args":{"url":"https://..."},                        "reason":"..."}
  {"action":"switch_tab",       "args":{"index":0},                                  "reason":"..."}
  {"action":"close_tab",        "args":{},                                            "reason":"..."}
  {"action":"done",             "args":{"message":"Task completed: ..."},            "reason":"..."}
  {"action":"error",            "args":{"message":"Cannot complete because: ..."},   "reason":"..."}

Rules:
- ALWAYS prefer click_index/fill_index over text/selector matching
- The elements list shows [N]<tag ...>text</tag> — use the N number for index actions
- Use {{key}} placeholders for credentials from memory
- After form submit always wait 2000ms
- Return 1-5 actions per round; you will be called again with fresh page state
- If task is done, return done action. If impossible, return error action.
"""

# ── LLM call (Phase 1: structured JSON, Phase 2: vision) ─────────────────────
def call_llm(model: str, task: str, memory: dict, page_info: dict,
             history: list, use_vision: bool = False) -> list:
    safe_mem = {
        k: ("***" if any(s in k.lower() for s in
            ["password","pass","pin","secret","card","token","key","ssn","cvv"]) else v)
        for k, v in memory.items()
    }

    page_ctx = {
        "url":      page_info.get("url"),
        "title":    page_info.get("title"),
        "text":     page_info.get("text", "")[:1500],
        "elements": page_info.get("elements", []),
    }

    if use_vision and page_info.get("screenshot_b64"):
        user_content = [
            {
                "type": "text",
                "text": json.dumps({
                    "task":    task,
                    "memory":  safe_mem,
                    "page":    page_ctx,
                    "history": history,
                }, indent=2)
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{page_info['screenshot_b64']}"
                }
            }
        ]
    else:
        user_content = json.dumps({
            "task":    task,
            "memory":  safe_mem,
            "page":    page_ctx,
            "history": history,
        }, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    resp = ollama.chat(
        model=model,
        messages=messages,
        format="json",
        options=GPU_CFG,
    )
    raw = resp["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("actions", parsed.get("steps", list(parsed.values())[0]))
    if not isinstance(parsed, list):
        raise ValueError("LLM did not return a JSON array")
    return parsed

# ── Action executor ───────────────────────────────────────────────────────────
NAV_TIMEOUT     = 30_000
VISIBLE_TIMEOUT = 30_000
CLICK_TIMEOUT   = 15_000
FILL_TIMEOUT    = 30_000

async def execute_action(page, action: dict, memory: dict,
                         context=None, emit_fn=None) -> str:
    act    = action.get("action", "")
    args   = action.get("args", {})
    reason = action.get("reason", "")

    def log(msg):
        agent_log.info(msg)
        if emit_fn:
            emit_fn("agent_log", {"msg": msg})

    log(f"ACTION: {act} | args={args} | reason={reason}")

    try:
        # ── Phase 1: index-based ──────────────────────────────────────────────
        if act == "click_index":
            return await click_by_index(page, int(args["index"]))

        elif act == "fill_index":
            val = inject_memory(args.get("value", ""), memory)
            return await fill_by_index(page, int(args["index"]), val)

        # ── Phase 3: multi-tab ────────────────────────────────────────────────
        elif act == "new_tab":
            url = args.get("url", "about:blank")
            new_page = await context.new_page()
            await new_page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            return f"Opened new tab: {url}"

        elif act == "switch_tab":
            tab_idx = int(args.get("index", 0))
            pages = context.pages
            if tab_idx < len(pages):
                await pages[tab_idx].bring_to_front()
                return f"Switched to tab {tab_idx}"
            return f"Tab {tab_idx} not found (have {len(pages)})"

        elif act == "close_tab":
            await page.close()
            pages = context.pages
            if pages:
                await pages[-1].bring_to_front()
            return "Closed current tab"

        # ── Standard actions ──────────────────────────────────────────────────
        elif act == "navigate":
            url = args["url"]
            await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
            await page.wait_for_timeout(1500)
            return f"Navigated to {url}"

        elif act == "click":
            txt = args["text"]
            loc = page.get_by_text(txt, exact=False).first
            await loc.wait_for(state="visible", timeout=VISIBLE_TIMEOUT)
            await loc.click(timeout=CLICK_TIMEOUT)
            await page.wait_for_timeout(1000)
            return f"Clicked text: {txt}"

        elif act == "click_sel":
            sel = args["selector"]
            await page.wait_for_selector(sel, state="visible", timeout=VISIBLE_TIMEOUT)
            await page.click(sel, timeout=CLICK_TIMEOUT)
            await page.wait_for_timeout(1000)
            return f"Clicked selector: {sel}"

        elif act == "fill":
            sel = args["selector"]
            val = inject_memory(args.get("value", ""), memory)
            await robust_fill(page, sel, val, timeout=FILL_TIMEOUT)
            return f"Filled {sel}"

        elif act == "fill_label":
            label = args["label"]
            val   = inject_memory(args.get("value", ""), memory)
            await robust_fill_by_label(page, label, val, timeout=FILL_TIMEOUT)
            return f"Filled field labeled '{label}'"

        elif act == "fill_placeholder":
            ph  = args["placeholder"]
            val = inject_memory(args.get("value", ""), memory)
            loc = page.get_by_placeholder(ph, exact=False).first
            await loc.wait_for(state="visible", timeout=FILL_TIMEOUT)
            await loc.fill(val, timeout=FILL_TIMEOUT)
            return f"Filled placeholder '{ph}'"

        elif act == "select_option":
            sel = args["selector"]
            val = args.get("value") or args.get("label")
            await page.select_option(sel, value=val, timeout=FILL_TIMEOUT)
            return f"Selected option '{val}' in {sel}"

        elif act == "press":
            key = args.get("key", "Enter")
            await page.keyboard.press(key)
            await page.wait_for_timeout(1000)
            return f"Pressed {key}"

        elif act == "wait":
            ms = int(args.get("ms", 2000))
            await page.wait_for_timeout(ms)
            return f"Waited {ms}ms"

        elif act == "wait_selector":
            sel = args["selector"]
            t   = int(args.get("timeout", 15000))
            await page.wait_for_selector(sel, state="visible", timeout=t)
            return f"Selector appeared: {sel}"

        elif act == "scroll":
            direction = args.get("direction", "down")
            amount    = int(args.get("amount", 500))
            dy = amount if direction == "down" else -amount
            await page.mouse.wheel(0, dy)
            await page.wait_for_timeout(500)
            return f"Scrolled {direction} {amount}px"

        elif act == "screenshot":
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(LOGS_DIR / f"screenshot_{ts}.png")
            await page.screenshot(path=path, full_page=False)
            log(f"Screenshot saved: {path}")
            if emit_fn:
                emit_fn("screenshot", {"path": os.path.basename(path)})
            return f"Screenshot: {path}"

        elif act == "done":
            return "DONE: " + args.get("message", "Task complete")

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
async def run_task(task: str, model: str = "llama3.2",
                   headless: bool = False, emit_fn=None,
                   use_vision: bool = False) -> dict:
    memory  = load_memory()
    history = []
    final   = {"status": "unknown", "message": "", "screenshots": []}

    def log(msg):
        agent_log.info(msg)
        if emit_fn:
            emit_fn("agent_log", {"msg": msg})

    log(f"=== NEW TASK: {task} | model={model} | vision={use_vision} ===")

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

        for round_n in range(1, 16):   # Phase 3: 15 rounds
            log(f"--- Round {round_n}/15 ---")

            trimmed_history = summarize_history(history, model, window=6)
            page_info = await get_page_info(page, capture_screenshot=use_vision)
            log(f"Page: {page_info['url']} | {page_info['title']} | {page_info.get('elem_count',0)} elements indexed")

            try:
                actions = call_llm(model, task, memory, page_info, trimmed_history, use_vision)
            except Exception as e:
                msg = f"LLM error: {e}"
                error_log.error(msg)
                log(f"LLM ERROR: {e}")
                final = {"status": "error", "message": msg}
                break

            done = False
            for action in actions:
                result = await execute_action(page, action, memory, context, emit_fn)

                pages = context.pages
                if pages:
                    page = pages[-1]

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
            final = {"status": "timeout", "message": "Max rounds (15) reached without completion"}
            error_log.error(final["message"])

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
