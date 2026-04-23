"""
BrowserAgent — Uses Ollama (local LLM) to plan and execute browser tasks
via Playwright. Every action is logged; errors go to logs/errors.log.
"""
import asyncio, json, logging, re, time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Page
import ollama

BASE_DIR   = Path(__file__).parent
MEMORY_DIR = BASE_DIR / "memory"
LOGS_DIR   = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)

# ── Loggers ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(LOGS_DIR / "agent.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
error_logger = logging.getLogger("errors")
_eh = logging.FileHandler(str(LOGS_DIR / "errors.log"))
_eh.setLevel(logging.ERROR)
error_logger.addHandler(_eh)


# ── Memory ────────────────────────────────────────────────────────────────────
def load_memory() -> dict:
    p = MEMORY_DIR / "user_memory.json"
    return json.loads(p.read_text()) if p.exists() else {}

def save_memory(data: dict):
    (MEMORY_DIR / "user_memory.json").write_text(json.dumps(data, indent=2))

def memory_to_context(mem: dict) -> str:
    if not mem:
        return "No user memory stored yet."
    return "User memory:\n" + "\n".join(f"  {k}: {v}" for k, v in mem.items())


# ── Action executor ────────────────────────────────────────────────────────────
async def execute_action(page: Page, action: dict, emit_fn=None) -> str:
    act  = action.get("action", "").lower()
    args = action.get("args", {})

    def _emit(msg):
        logging.info(msg)
        if emit_fn:
            emit_fn("log", {"msg": msg, "ts": datetime.now().strftime("%H:%M:%S")})

    try:
        if act == "navigate":
            url = args.get("url", "")
            if not url.startswith("http"):
                url = "https://" + url
            _emit(f"Navigating to {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)
            return f"Navigated to {url}"

        elif act == "click":
            sel  = args.get("selector", "")
            text = args.get("text", "")
            if text:
                _emit(f"Clicking element with text '{text}'")
                await page.get_by_text(text, exact=False).first.click(timeout=10000)
            else:
                _emit(f"Clicking selector '{sel}'")
                await page.locator(sel).first.click(timeout=10000)
            await page.wait_for_timeout(800)
            return f"Clicked {text or sel}"

        elif act == "type":
            sel  = args.get("selector", "")
            text = args.get("text", "")
            _emit(f"Typing into '{sel}'")
            loc = page.locator(sel).first
            if args.get("clear", True):
                await loc.clear()
            await loc.type(text, delay=40)
            return f"Typed into {sel}"

        elif act == "fill":
            sel = args.get("selector", "")
            val = args.get("value", "")
            _emit(f"Filling '{sel}'")
            await page.locator(sel).first.fill(val)
            return f"Filled {sel}"

        elif act == "press":
            key = args.get("key", "Enter")
            _emit(f"Pressing {key}")
            await page.keyboard.press(key)
            await page.wait_for_timeout(800)
            return f"Pressed {key}"

        elif act == "wait":
            ms = int(args.get("ms", 2000))
            _emit(f"Waiting {ms}ms")
            await page.wait_for_timeout(ms)
            return f"Waited {ms}ms"

        elif act == "wait_for_selector":
            sel = args.get("selector", "")
            _emit(f"Waiting for selector '{sel}'")
            await page.wait_for_selector(sel, timeout=15000)
            return f"Selector appeared: {sel}"

        elif act == "scroll":
            direction = args.get("direction", "down")
            amount    = int(args.get("amount", 500))
            dy = amount if direction == "down" else -amount
            _emit(f"Scrolling {direction} {amount}px")
            await page.evaluate(f"window.scrollBy(0, {dy})")
            return f"Scrolled {direction}"

        elif act == "screenshot":
            path = str(LOGS_DIR / f"ss_{int(time.time())}.png")
            await page.screenshot(path=path)
            _emit(f"Screenshot: {path}")
            return f"Screenshot: {path}"

        elif act == "get_text":
            sel = args.get("selector", "body")
            _emit(f"Getting text from '{sel}'")
            text = await page.locator(sel).first.inner_text()
            return f"TEXT: {text[:1000]}"

        elif act == "select_option":
            sel = args.get("selector", "")
            val = args.get("value", "")
            _emit(f"Selecting '{val}' from '{sel}'")
            await page.locator(sel).first.select_option(value=val)
            return f"Selected {val}"

        elif act == "hover":
            sel = args.get("selector", "")
            _emit(f"Hovering over '{sel}'")
            await page.locator(sel).first.hover()
            return f"Hovered {sel}"

        elif act == "done":
            msg = args.get("message", "Task completed.")
            _emit(f"DONE: {msg}")
            return f"DONE: {msg}"

        elif act == "error":
            msg = args.get("message", "Unknown error.")
            _emit(f"ERROR: {msg}")
            error_logger.error(msg)
            return f"ERROR: {msg}"

        else:
            _emit(f"Unknown action: {act}")
            return f"Unknown action: {act}"

    except Exception as e:
        err = f"Action '{act}' failed: {e}"
        error_logger.error(err, exc_info=True)
        _emit(f"ERROR: {err}")
        return f"ERROR: {err}"


# ── LLM ───────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are BrowserAgent, an AI that controls a web browser.

You receive the user task, memory context, and current page state.
Respond ONLY with a JSON array of browser actions. No prose, no explanation outside JSON.

Action schema:
{ "action": "<name>", "args": { ... }, "reason": "why" }

Available actions:
  navigate           args: { url }
  click              args: { text } or { selector }
  type               args: { selector, text, clear:true }
  fill               args: { selector, value }
  press              args: { key }
  wait               args: { ms }
  wait_for_selector  args: { selector }
  scroll             args: { direction, amount }
  screenshot         args: {}
  get_text           args: { selector }
  select_option      args: { selector, value }
  hover              args: { selector }
  done               args: { message }
  error              args: { message }

Rules:
- Return ONLY a valid JSON array.
- Use memory for all credentials and personal data (logins, addresses, cards).
- Always end your action list with done or error.
- Take a screenshot after each major step.
- Wait 1500ms after every navigation.
- Prefer text-based click selectors. Use CSS selectors as fallback.
"""

async def get_page_info(page: Page) -> dict:
    try:
        url   = page.url
        title = await page.title()
        text  = await page.evaluate("document.body ? document.body.innerText.slice(0,3000) : ''")
        html  = await page.evaluate("""() => {
            const out = [];
            document.querySelectorAll('form,input,button,select,a,textarea,[role="button"]').forEach(el => {
                const a = {};
                for (const attr of el.attributes) a[attr.name] = attr.value;
                out.push({ tag: el.tagName.toLowerCase(), attrs: a, text: (el.innerText||'').slice(0,60) });
            });
            return JSON.stringify(out.slice(0,80), null, 2);
        }""")
        return {"url": url, "title": title, "text": text, "html_snippet": html}
    except Exception:
        return {"url": "unknown", "title": "unknown", "text": "", "html_snippet": ""}


def call_llm(model: str, task: str, memory: dict, page_info: dict,
             history: list, emit_fn=None) -> list:
    page_ctx = (
        f"URL: {page_info['url']}\n"
        f"Title: {page_info['title']}\n"
        f"Visible text:\n{page_info['text'][:2500]}\n\n"
        f"Interactive elements (JSON):\n{page_info['html_snippet'][:2500]}"
    )
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += history[-10:]
    msgs.append({"role": "user", "content": (
        f"TASK: {task}\n\n"
        f"MEMORY:\n{memory_to_context(memory)}\n\n"
        f"PAGE STATE:\n{page_ctx}\n\n"
        "Return JSON array of next actions."
    )})

    if emit_fn:
        emit_fn("log", {"msg": f"Planning next actions with {model}...",
                        "ts": datetime.now().strftime("%H:%M:%S")})
    resp = ollama.chat(model=model, messages=msgs)
    raw  = resp["message"]["content"].strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()

    try:
        actions = json.loads(raw)
        return [actions] if isinstance(actions, dict) else actions
    except Exception as e:
        error_logger.error(f"JSON parse failed: {e} | raw: {raw[:300]}")
        return [{"action": "error",
                 "args": {"message": f"LLM returned invalid JSON: {raw[:200]}"}}]


# ── Main runner ────────────────────────────────────────────────────────────────
async def run_task(task: str, model: str = "llama3.2",
                   headless: bool = False, emit_fn=None) -> dict:
    memory   = load_memory()
    history  = []
    results  = []
    MAX_ROUNDS = 12

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-blink-features=AutomationControlled"]
        )
        ctx  = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()
        if emit_fn:
            emit_fn("log", {"msg": "Browser launched (Chromium)",
                            "ts": datetime.now().strftime("%H:%M:%S")})

        for rnd in range(MAX_ROUNDS):
            page_info = await get_page_info(page)
            actions   = call_llm(model, task, memory, page_info, history, emit_fn)

            round_results = []
            done = False
            for action in actions:
                result = await execute_action(page, action, emit_fn)
                round_results.append({"action": action, "result": result})
                results.append(result)
                if result.startswith("DONE:") or result.startswith("ERROR:"):
                    done = True
                    break

            history.append({"role": "assistant", "content": json.dumps(actions)})
            history.append({"role": "user",
                            "content": f"Action results: {json.dumps(round_results)}"})
            if done:
                break

        final_ss = str(LOGS_DIR / f"final_{int(time.time())}.png")
        try:
            await page.screenshot(path=final_ss, full_page=False)
        except Exception:
            final_ss = None
        await browser.close()

    return {"results": results, "screenshot": final_ss}
