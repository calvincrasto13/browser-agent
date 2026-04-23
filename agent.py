"""
BrowserAgent v3 — GPU-accelerated + Comet-style browser control.

Comet-style control:
  - Every round captures BOTH a screenshot AND a DOM accessibility snapshot
  - The DOM snapshot gives the LLM a structured map of every interactive
    element with stable selectors (role, name, id, aria-label, coords)
  - LLM can act on DOM elements by index (#1, #2 …) rather than guessing
    CSS selectors — same approach used by Perplexity Comet & browser-use
  - Falls back to vision-only when DOM is unavailable (SPAs, canvas pages)

GPU acceleration:
  - Reads gpu_config.json written by setup.py
  - Passes num_gpu, use_mmap, use_mlock to every Ollama call
  - Pins the model in VRAM between rounds (keep_alive=600s)
  - Parallel screenshot + DOM capture via asyncio.gather
"""
import asyncio, base64, json, logging, re, time
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


# ── GPU config ────────────────────────────────────────────────────────────────
def load_gpu_config() -> dict:
    p = BASE_DIR / "gpu_config.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"num_gpu_layers": 0, "use_mmap": True, "use_mlock": False, "gpu_type": "cpu", "vram_gb": 0}

GPU_CFG = load_gpu_config()

def gpu_options() -> dict:
    """Build Ollama options dict for GPU acceleration."""
    opts = {
        "num_gpu":  GPU_CFG.get("num_gpu_layers", 0),
        "use_mmap": GPU_CFG.get("use_mmap", True),
        "num_ctx":  8192,   # large context for DOM snapshots
        "temperature": 0.1, # low temp = more deterministic actions
    }
    if GPU_CFG.get("use_mlock"):
        opts["use_mlock"] = True
    return opts


# ── Credential memory ─────────────────────────────────────────────────────────
def load_memory() -> dict:
    p = MEMORY_DIR / "user_memory.json"
    return json.loads(p.read_text()) if p.exists() else {}

def save_memory(data: dict):
    (MEMORY_DIR / "user_memory.json").write_text(json.dumps(data, indent=2))

def memory_to_context(mem: dict) -> str:
    if not mem:
        return "No user memory stored yet."
    return "User credentials/memory:\n" + "\n".join(
        f"  {k}: {v}" for k, v in mem.items() if not k.startswith("_")
    )


# ── Screenshot — base64 in RAM ────────────────────────────────────────────────
async def screenshot_b64(page: Page) -> str:
    png_bytes = await page.screenshot(type="png")
    return base64.b64encode(png_bytes).decode("utf-8")


# ── Comet-style DOM snapshot ──────────────────────────────────────────────────
async def dom_snapshot(page: Page) -> str:
    """
    Build a compact, numbered list of every interactive element on the page.
    Format mirrors Perplexity Comet / browser-use accessibility tree:

      #1  [button]  "Sign In"            id=signin-btn        (920, 48)
      #2  [input]   "Search"             id=search-box        (640, 120)
      #3  [link]    "Weekly Flyer"       href=/flyer          (200, 300)

    The LLM references elements as #N in its actions.
    """
    try:
        elements = await page.evaluate("""
        () => {
            const TAGS = ['a','button','input','select','textarea',
                          '[role="button"]','[role="link"]','[role="menuitem"]',
                          '[role="tab"]','[role="checkbox"]','[role="radio"]',
                          '[role="combobox"]','[role="option"]'];
            const seen  = new Set();
            const items = [];
            let   idx   = 1;

            for (const sel of TAGS) {
                for (const el of document.querySelectorAll(sel)) {
                    if (seen.has(el)) continue;
                    seen.add(el);

                    const rect = el.getBoundingClientRect();
                    if (rect.width < 2 || rect.height < 2) continue;  // invisible

                    const cx = Math.round(rect.left + rect.width  / 2);
                    const cy = Math.round(rect.top  + rect.height / 2);

                    const label =
                        el.getAttribute('aria-label') ||
                        el.getAttribute('placeholder') ||
                        el.getAttribute('title') ||
                        el.getAttribute('alt') ||
                        (el.innerText || '').trim().slice(0, 60) ||
                        el.getAttribute('name') ||
                        el.getAttribute('id') ||
                        '';

                    const role = el.getAttribute('role') || el.tagName.toLowerCase();
                    const id   = el.id   ? `id=${el.id}`   : '';
                    const href = el.href ? `href=${new URL(el.href).pathname}` : '';
                    const meta = [id, href].filter(Boolean).join(' ');

                    items.push({
                        idx,
                        role,
                        label: label.replace(/\n/g,' '),
                        meta,
                        cx, cy,
                    });
                    idx++;
                    if (idx > 150) break;  // cap at 150 elements
                }
            }
            return items;
        }
        """)

        lines = []
        for el in elements:
            meta = f"  {el['meta']}" if el['meta'] else ""
            lines.append(
                f"  #{el['idx']:>3}  [{el['role']:<12}]  \"{el['label']:<50}"  \
                f"{meta}  ({el['cx']},{el['cy']})"
            )
        header = f"DOM elements ({len(elements)} interactive, page: {page.url})\n"
        return header + "\n".join(lines)

    except Exception as e:
        error_logger.error(f"DOM snapshot failed: {e}")
        return f"DOM snapshot unavailable: {e}"


# ── Robust click: text + 5 fallback strategies ────────────────────────────────
async def robust_click_text(page: Page, text: str, log_fn=None) -> str:
    tried = []
    for strategy, fn in [
        ("get_by_text",    lambda: page.get_by_text(text, exact=False).first.click(timeout=5000)),
        ("role=button",    lambda: page.get_by_role("button", name=re.compile(text, re.I)).first.click(timeout=5000)),
        ("role=link",      lambda: page.get_by_role("link",   name=re.compile(text, re.I)).first.click(timeout=5000)),
        ("aria-label",     lambda: page.locator(f'[aria-label*="{text}" i]').first.click(timeout=5000)),
        ("XPath",          lambda: page.locator(f'xpath=//*[contains(translate(normalize-space(.),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"),"{text.lower()}")]').first.click(timeout=5000)),
    ]:
        tried.append(strategy)
        try:
            await fn()
            return f"Clicked ({strategy}): {text}"
        except Exception:
            pass

    # Cross-frame JS fallback
    for frame in page.frames:
        try:
            ok = await frame.evaluate(
                """(t) => { for(const el of document.querySelectorAll('button,a,[role=button],[role=link]')){
                    if((el.innerText||el.textContent||'').toLowerCase().includes(t)){el.click();return true;}} return false;}""",
                text.lower()
            )
            if ok:
                return f"Clicked (frame JS): {text}"
        except Exception:
            pass

    raise RuntimeError(f"All click strategies failed for '{text}'. Tried: {tried}")


# ── Action executor ────────────────────────────────────────────────────────────
async def execute_action(page: Page, action: dict, emit_fn=None) -> str:
    act  = action.get("action", "").lower()
    args = action.get("args", {})

    def _emit(msg):
        logging.info(msg)
        if emit_fn:
            emit_fn("log", {"msg": msg, "ts": datetime.now().strftime("%H:%M:%S")})

    try:
        # ── Comet-style: click by DOM index ─────────────────────────────────
        if act == "click_element":
            idx = int(args.get("index", 0))
            _emit(f"Clicking DOM element #{idx} (Comet-style)")
            result = await page.evaluate(f"""
            () => {{
                const TAGS = ['a','button','input','select','textarea',
                              '[role=\'button\']','[role=\'link\']','[role=\'menuitem\']',
                              '[role=\'tab\']','[role=\'checkbox\']','[role=\'radio\']',
                              '[role=\'combobox\']','[role=\'option\']'];
                const seen = new Set();
                let   i    = 0;
                for (const sel of TAGS) {{
                    for (const el of document.querySelectorAll(sel)) {{
                        if (seen.has(el)) continue;
                        seen.add(el);
                        const r = el.getBoundingClientRect();
                        if (r.width < 2 || r.height < 2) continue;
                        i++;
                        if (i === {idx}) {{ el.click(); return true; }}
                    }}
                }}
                return false;
            }}
            """)
            await page.wait_for_timeout(800)
            return f"Clicked DOM element #{idx}" if result else f"ERROR: DOM element #{idx} not found"

        # ── Comet-style: fill by DOM index ───────────────────────────────────
        elif act == "fill_element":
            idx = int(args.get("index", 0))
            val = args.get("value", "")
            _emit(f"Filling DOM element #{idx} with value")
            result = await page.evaluate(f"""
            () => {{
                const TAGS = ['input','textarea','select','[contenteditable]'];
                const seen = new Set();
                let   i    = 0;
                const allInputs = document.querySelectorAll(TAGS.join(','));
                for (const el of allInputs) {{
                    const r = el.getBoundingClientRect();
                    if (r.width < 2 || r.height < 2) continue;
                    i++;
                    if (i === {idx}) {{
                        el.focus();
                        el.value = '';
                        el.dispatchEvent(new Event('input', {{bubbles:true}}));
                        return true;
                    }}
                }}
                return false;
            }}
            """)
            if result:
                # Find the focused element and type into it
                await page.keyboard.type(val, delay=40)
            await page.wait_for_timeout(400)
            return f"Filled DOM element #{idx}" if result else f"ERROR: Input #{idx} not found"

        # ── navigate ─────────────────────────────────────────────────────────
        elif act == "navigate":
            url = args.get("url", "")
            if not url.startswith("http"):
                url = "https://" + url
            _emit(f"Navigating to {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)
            return f"Navigated to {url}"

        # ── click (text / selector) ──────────────────────────────────────────
        elif act == "click":
            sel  = args.get("selector", "")
            text = args.get("text", "")
            if text:
                _emit(f"Clicking '{text}' (robust fallback chain)")
                result = await robust_click_text(page, text, log_fn=_emit)
                await page.wait_for_timeout(800)
                return result
            else:
                _emit(f"Clicking selector '{sel}'")
                await page.locator(sel).first.click(timeout=10000)
                await page.wait_for_timeout(800)
                return f"Clicked {sel}"

        # ── click_coords ─────────────────────────────────────────────────────
        elif act == "click_coords":
            x = int(args.get("x", 0))
            y = int(args.get("y", 0))
            _emit(f"Clicking at ({x}, {y})")
            await page.mouse.click(x, y)
            await page.wait_for_timeout(800)
            return f"Clicked at ({x}, {y})"

        # ── type ─────────────────────────────────────────────────────────────
        elif act == "type":
            sel  = args.get("selector", "")
            text = args.get("text", "")
            _emit(f"Typing into '{sel}'")
            loc = page.locator(sel).first
            if args.get("clear", True):
                await loc.clear()
            await loc.type(text, delay=40)
            return f"Typed into {sel}"

        # ── fill ─────────────────────────────────────────────────────────────
        elif act == "fill":
            sel = args.get("selector", "")
            val = args.get("value", "")
            _emit(f"Filling '{sel}'")
            await page.locator(sel).first.fill(val)
            return f"Filled {sel}"

        # ── press ─────────────────────────────────────────────────────────────
        elif act == "press":
            key = args.get("key", "Enter")
            _emit(f"Pressing {key}")
            await page.keyboard.press(key)
            await page.wait_for_timeout(800)
            return f"Pressed {key}"

        # ── scroll ────────────────────────────────────────────────────────────
        elif act == "scroll":
            direction = args.get("direction", "down")
            amount    = int(args.get("amount", 500))
            dy = amount if direction == "down" else -amount
            _emit(f"Scrolling {direction} {amount}px")
            await page.evaluate(f"window.scrollBy(0, {dy})")
            return f"Scrolled {direction}"

        # ── wait ──────────────────────────────────────────────────────────────
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

        # ── get_text ──────────────────────────────────────────────────────────
        elif act == "get_text":
            sel = args.get("selector", "body")
            _emit(f"Getting text from '{sel}'")
            text = await page.locator(sel).first.inner_text()
            return f"TEXT: {text[:1000]}"

        # ── select_option ─────────────────────────────────────────────────────
        elif act == "select_option":
            sel = args.get("selector", "")
            val = args.get("value", "")
            _emit(f"Selecting '{val}' from '{sel}'")
            await page.locator(sel).first.select_option(value=val)
            return f"Selected {val}"

        # ── hover ─────────────────────────────────────────────────────────────
        elif act == "hover":
            sel = args.get("selector", "")
            _emit(f"Hovering '{sel}'")
            await page.locator(sel).first.hover()
            return f"Hovered {sel}"

        # ── done / error ──────────────────────────────────────────────────────
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
SYSTEM_PROMPT = """You are BrowserAgent, an AI that controls a web browser.

Each round you receive:
  1. A screenshot of the current browser state (look at it carefully)
  2. A DOM element map — every interactive element numbered #1, #2, #3 …
  3. The user task, credentials/memory, working memory, and last action results

Respond ONLY with a single valid JSON object. No prose outside JSON.

Response schema:
{
  "observation": "What you see on screen AND in the DOM map",
  "reasoning":   "Why you are taking the next actions",
  "memory_update": {
    "steps_done":    ["completed steps"],
    "current_state": "where you are in the task",
    "blockers":      ["any issues"]
  },
  "actions": [
    { "action": "<name>", "args": { ... }, "reason": "why" }
  ]
}

Available actions (PREFER DOM-index actions — they are faster and more reliable):

  COMET-STYLE (preferred — use DOM element numbers from the map):
    click_element  args: { index }          ← click element #N from DOM map
    fill_element   args: { index, value }   ← fill input #N from DOM map

  VISION-FALLBACK (use when element not in DOM map, or on canvas/SPA pages):
    click          args: { text } or { selector }
    click_coords   args: { x, y }           ← pixel coords visible on screenshot
    fill           args: { selector, value }
    type           args: { selector, text, clear:true }

  NAVIGATION & INTERACTION:
    navigate          args: { url }
    press             args: { key }
    scroll            args: { direction, amount }
    wait              args: { ms }
    wait_for_selector args: { selector }
    get_text          args: { selector }
    select_option     args: { selector, value }
    hover             args: { selector }
    done              args: { message }
    error             args: { message }

CRITICAL RULES:
- ALWAYS use click_element / fill_element when the element appears in the DOM map.
- Only use click { text } or click_coords as fallback when not in DOM map.
- Use credential memory for all logins — never guess passwords.
- Describe what you SEE in observation before deciding.
- STOP and call done as soon as the task goal is visibly achieved.
- If CAPTCHA or MFA appears, call error immediately.
- Never retry the exact same failed action twice — switch strategy.
- End every response with either done or error.
"""


# ── Vision LLM call (GPU-accelerated) ────────────────────────────────────────
def call_vision_llm(
    model: str,
    task: str,
    credential_memory: dict,
    working_memory: dict,
    screenshot_b64_str: str,
    dom_text: str,
    last_results: list,
    emit_fn=None,
) -> dict:
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64_str}"},
        },
        {
            "type": "text",
            "text": (
                f"TASK: {task}\n\n"
                f"CREDENTIALS/MEMORY:\n{memory_to_context(credential_memory)}\n\n"
                f"DOM ELEMENT MAP:\n{dom_text}\n\n"
                f"WORKING MEMORY:\n{json.dumps(working_memory, indent=2)}\n\n"
                f"LAST ACTION RESULTS:\n"
                + ("\n".join(last_results) if last_results else "None yet.") +
                "\n\nAnalyse the screenshot + DOM map, then return your JSON response."
            ),
        },
    ]

    if emit_fn:
        gpu_label = GPU_CFG.get("gpu_name", "CPU")
        emit_fn("log", {"msg": f"👁️  Sending to {model} [{gpu_label}] …",
                        "ts": datetime.now().strftime("%H:%M:%S")})

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        options=gpu_options(),
        keep_alive=600,   # pin model in VRAM for 10 min between rounds
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
        error_logger.error(f"LLM JSON parse failed: {e} | raw: {raw[:300]}")
        return {
            "observation":   "Could not parse LLM response.",
            "reasoning":     "JSON parse error.",
            "memory_update": {},
            "actions": [{"action": "error", "args": {"message": f"LLM returned invalid JSON: {raw[:200]}"}}],
        }


# ── Warm up model (load into VRAM before first task) ─────────────────────────
def warmup_model(model: str):
    try:
        ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "ready"}],
            options=gpu_options(),
            keep_alive=600,
        )
        logging.info(f"Model {model} warmed up in VRAM")
    except Exception as e:
        logging.warning(f"Model warmup failed (non-fatal): {e}")


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
    gpu_label = GPU_CFG.get("gpu_name", "CPU")
    vram      = GPU_CFG.get("vram_gb", 0)
    layers    = GPU_CFG.get("num_gpu_layers", 0)
    _log(f"GPU: {gpu_label} | VRAM: {vram} GB | Layers offloaded: {layers}")

    credential_memory = load_memory()
    working_memory: dict = {"steps_done": [], "current_state": "Starting", "blockers": []}
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

            # Parallel capture: screenshot + DOM snapshot simultaneously
            try:
                ss, dom_text = await asyncio.gather(
                    screenshot_b64(page),
                    dom_snapshot(page),
                )
            except Exception as e:
                _log(f"Capture failed: {e}")
                ss, dom_text = "", "DOM unavailable"

            llm_resp = call_vision_llm(
                model=model,
                task=task,
                credential_memory=credential_memory,
                working_memory=working_memory,
                screenshot_b64_str=ss,
                dom_text=dom_text,
                last_results=last_results,
                emit_fn=emit_fn,
            )

            del ss  # free RAM immediately

            _emit("vision", {
                "round":       round_num,
                "observation": llm_resp.get("observation", ""),
                "reasoning":   llm_resp.get("reasoning",   ""),
                "memory":      llm_resp.get("memory_update", {}),
                "dom_elements": dom_text.splitlines()[0] if dom_text else "",
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

        # Save final screenshot
        ss_name = ""
        try:
            ss_path = str(LOGS_DIR / f"final_{int(time.time())}.png")
            await page.screenshot(path=ss_path)
            ss_name = Path(ss_path).name
            _log(f"Final screenshot: {ss_name}")
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
