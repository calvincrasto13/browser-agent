"""
BrowserAgent v2 — Vision-guided browser automation.

Each round:
  1. Take a screenshot → encode as base64 in RAM
  2. Send screenshot + context to a vision-capable Ollama model
  3. LLM returns: observation, reasoning, memory_update, actions
  4. Execute actions via Playwright
  5. Discard screenshot — never written to disk
  6. Update in-RAM working memory
Only the final screenshot is saved to logs/.
"""
import asyncio, base64, json, logging, re, time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Page, Frame
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


# ── Credential memory (persisted) ─────────────────────────────────────────────
def load_memory() -> dict:
    p = MEMORY_DIR / "user_memory.json"
    return json.loads(p.read_text()) if p.exists() else {}

def save_memory(data: dict):
    (MEMORY_DIR / "user_memory.json").write_text(json.dumps(data, indent=2))

def memory_to_context(mem: dict) -> str:
    if not mem:
        return "No user memory stored yet."
    return "User credentials/memory:\n" + "\n".join(
        f"  {k}: {v}" for k, v in mem.items()
        if not k.startswith("_")
    )


# ── Screenshot helper — base64 in RAM, never written to disk ──────────────────
async def screenshot_b64(page: Page) -> str:
    png_bytes = await page.screenshot(type="png")
    return base64.b64encode(png_bytes).decode("utf-8")


# ── Robust click: 5-strategy fallback chain ─────────────────────────────────────
async def robust_click_text(page: Page, text: str, log_fn=None) -> str:
    """
    Try clicking an element matching `text` using 5 escalating strategies.
    Handles shadow DOM, cross-frame buttons (e.g. 'Continue with Google' on LinkedIn).
    Returns a result string. Raises only if all strategies fail.
    """
    tried = []

    # Strategy 1: Standard get_by_text
    try:
        tried.append("get_by_text")
        await page.get_by_text(text, exact=False).first.click(timeout=5000)
        return f"Clicked (get_by_text): {text}"
    except Exception:
        pass

    # Strategy 2: Semantic role — button or link with matching name
    for role in ("button", "link"):
        try:
            tried.append(f"get_by_role({role})")
            await page.get_by_role(role, name=re.compile(text, re.IGNORECASE)).first.click(timeout=5000)
            return f"Clicked (get_by_role {role}): {text}"
        except Exception:
            pass

    # Strategy 3: ARIA label CSS attribute selector
    try:
        tried.append("aria-label selector")
        sel = f'[aria-label*="{text}" i]'
        await page.locator(sel).first.click(timeout=5000)
        return f"Clicked (aria-label): {text}"
    except Exception:
        pass

    # Strategy 4: Search all frames (iframes) for the text and click there
    try:
        tried.append("cross-frame JS")
        result = await _click_in_all_frames(page, text)
        if result:
            return f"Clicked (frame JS): {text}"
    except Exception:
        pass

    # Strategy 5: XPath innerText across main frame
    try:
        tried.append("XPath innerText")
        xpath = f'//*[contains(translate(normalize-space(.), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "{text.lower()}")]'
        await page.locator(f"xpath={xpath}").first.click(timeout=5000)
        return f"Clicked (XPath): {text}"
    except Exception:
        pass

    raise RuntimeError(f"All click strategies failed for '{text}'. Tried: {tried}")


async def _click_in_all_frames(page: Page, text: str) -> bool:
    """
    Walk every frame on the page (including nested iframes) and click
    the first element whose visible text matches. Handles LinkedIn-style
    social login buttons embedded inside cross-origin iframes.
    """
    frames = page.frames
    text_lower = text.lower()
    for frame in frames:
        try:
            clicked = await frame.evaluate(
                """(textLower) => {
                    const els = document.querySelectorAll('button, a, [role="button"], [role="link"], div, span');
                    for (const el of els) {
                        if ((el.innerText || el.textContent || '').toLowerCase().includes(textLower)) {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }""",
                text_lower,
            )
            if clicked:
                return True
        except Exception:
            continue
    return False


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
                _emit(f"Clicking element with text '{text}' (robust fallback chain)")
                result = await robust_click_text(page, text, log_fn=_emit)
                await page.wait_for_timeout(800)
                return result
            else:
                _emit(f"Clicking selector '{sel}'")
                await page.locator(sel).first.click(timeout=10000)
                await page.wait_for_timeout(800)
                return f"Clicked {sel}"

        elif act == "click_coords":
            x = int(args.get("x", 0))
            y = int(args.get("y", 0))
            _emit(f"Clicking at coordinates ({x}, {y})")
            await page.mouse.click(x, y)
            await page.wait_for_timeout(800)
            return f"Clicked at ({x}, {y})"

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


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are BrowserAgent, an AI that controls a web browser by visually reading screenshots.

Each round you receive:
  - A screenshot of the current browser state (analyse it carefully)
  - The user task
  - User credentials/memory
  - Your working memory from previous rounds
  - Results of your last actions

Respond ONLY with a single valid JSON object. No prose outside JSON.

Response schema:
{
  "observation": "What you see on screen right now",
  "reasoning":   "Why you are taking the next actions",
  "memory_update": {
    "steps_done":    ["list of steps completed so far"],
    "current_state": "where you are in the task",
    "blockers":      ["any issues encountered"]
  },
  "actions": [
    { "action": "<name>", "args": { ... }, "reason": "why" }
  ]
}

Available actions:
  navigate          args: { url }
  click             args: { text } or { selector }   ← automatically tries 5 fallback strategies including iframe/shadow DOM
  click_coords      args: { x, y }                  ← use pixel coords when you can see the button visually
  type              args: { selector, text, clear:true }
  fill              args: { selector, value }
  press             args: { key }
  wait              args: { ms }
  wait_for_selector args: { selector }
  scroll            args: { direction, amount }
  get_text          args: { selector }
  select_option     args: { selector, value }
  hover             args: { selector }
  done              args: { message }
  error             args: { message }

CRITICAL RULES:
- Always describe what you SEE on the screenshot in "observation" before acting.
- Use credential memory for all logins — never guess passwords.
- click { text } automatically tries get_by_text, get_by_role, aria-label, iframe search, and XPath — prefer it.
- If click { text } fails AND you can see the button on screen, immediately follow up with click_coords { x, y }.
- STOP and call done as soon as the task goal is visibly achieved.
- If you see a CAPTCHA or MFA screen, call error immediately — do not attempt to bypass.
- Never retry the exact same failed action twice in a row — switch strategy.
- End every response with either done or error in the actions list.
- Screenshots used for reasoning are discarded after this round — never stored.
"""


# ── Vision LLM call ───────────────────────────────────────────────────────────
def call_vision_llm(
    model: str,
    task: str,
    credential_memory: dict,
    working_memory: dict,
    screenshot_b64: str,
    last_results: list,
    emit_fn=None,
) -> dict:
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
        {
            "type": "text",
            "text": (
                f"TASK: {task}\n\n"
                f"CREDENTIALS/MEMORY:\n{memory_to_context(credential_memory)}\n\n"
                f"WORKING MEMORY (your progress so far):\n"
                f"{json.dumps(working_memory, indent=2)}\n\n"
                f"LAST ACTION RESULTS:\n"
                + ("\n".join(last_results) if last_results else "None yet — first round.") +
                "\n\nAnalyse the screenshot carefully, then return your JSON response."
            ),
        },
    ]

    if emit_fn:
        emit_fn("log", {"msg": f"👁️  Sending screenshot to {model} for analysis...",
                        "ts": datetime.now().strftime("%H:%M:%S")})

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
    )
    raw = resp["message"]["content"].strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()

    try:
        parsed = json.loads(raw)
        if "actions" not in parsed:
            parsed["actions"] = [{"action": "error", "args": {"message": "LLM returned no actions"}}]
        return parsed
    except Exception as e:
        error_logger.error(f"Vision LLM JSON parse failed: {e} | raw: {raw[:300]}")
        return {
            "observation":   "Could not parse LLM response.",
            "reasoning":     "JSON parse error.",
            "memory_update": {},
            "actions": [{"action": "error", "args": {"message": f"LLM returned invalid JSON: {raw[:200]}"}}],
        }


# ── Main runner ────────────────────────────────────────────────────────────────
async def run_task(
    task: str,
    model: str = "llama3.2-vision",
    headless: bool = False,
    max_rounds: int = 15,
    emit_fn=None,
) -> dict:
    def _emit(event, payload):
        if emit_fn:
            emit_fn(event, payload)

    def _log(msg):
        logging.info(msg)
        _emit("log", {"msg": msg, "ts": datetime.now().strftime("%H:%M:%S")})

    _log(f"Task started: {task}")
    credential_memory = load_memory()

    working_memory: dict = {
        "steps_done":    [],
        "current_state": "Starting",
        "blockers":      [],
    }

    start = time.time()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        _log("Browser launched (Chromium)")

        final_result = "Task incomplete"
        done         = False
        last_results: list = []

        for round_num in range(1, max_rounds + 1):
            if done:
                break

            _log(f"── Round {round_num}/{max_rounds} ──")

            try:
                ss = await screenshot_b64(page)
            except Exception as e:
                _log(f"Screenshot failed: {e}")
                ss = ""

            llm_resp = call_vision_llm(
                model=model,
                task=task,
                credential_memory=credential_memory,
                working_memory=working_memory,
                screenshot_b64=ss,
                last_results=last_results,
                emit_fn=emit_fn,
            )

            del ss  # discard — GC reclaims RAM

            _emit("vision", {
                "round":       round_num,
                "observation": llm_resp.get("observation", ""),
                "reasoning":   llm_resp.get("reasoning",   ""),
                "memory":      llm_resp.get("memory_update", {}),
            })

            if llm_resp.get("memory_update"):
                working_memory.update(llm_resp["memory_update"])

            last_results = []
            for action in llm_resp.get("actions", []):
                result = await execute_action(page, action, emit_fn=emit_fn)
                last_results.append(result)

                if result.startswith("DONE:"):
                    final_result = result[5:].strip()
                    done = True
                    break
                if result.startswith("ERROR:") and action.get("action") == "error":
                    final_result = result
                    done = True
                    break

        # Save ONE final screenshot to disk
        ss_name = ""
        try:
            ss_path = str(LOGS_DIR / f"final_{int(time.time())}.png")
            await page.screenshot(path=ss_path)
            ss_name = Path(ss_path).name
            _log(f"Final screenshot saved: {ss_name}")
        except Exception as e:
            _log(f"Final screenshot failed: {e}")

        elapsed = round(time.time() - start, 1)
        _log(f"Task complete in {elapsed}s")

        errors = []
        ef = LOGS_DIR / "errors.log"
        if ef.exists():
            lines = ef.read_text().splitlines()
            errors = [l for l in lines[-20:] if l.strip()]

        await browser.close()

        return {
            "result":         final_result,
            "elapsed":        elapsed,
            "screenshot":     ss_name,
            "errors":         errors,
            "working_memory": working_memory,
        }
