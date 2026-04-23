# BrowserAgent v2 — Vision-Guided Browser Automation 🤖👁️

A local desktop application where a vision-capable LLM **looks at screenshots** of a browser and decides the next action — just like a human would. No cloud APIs. Everything runs on your machine.

---

## How It Works (Vision Loop)

```
You type a task
       ↓
Agent opens Chromium browser
       ↓
┌──────────────────────────────────────────┐
│  Each Round (up to 15 by default):       │
│                                          │
│  1. Take screenshot (in-memory, base64)  │
│  2. Send screenshot + context to LLM    │
│  3. LLM visually reads the screen        │
│  4. LLM returns JSON: observation +      │
│     reasoning + memory_update + actions  │
│  5. Execute actions (click/fill/press…)  │
│  6. Discard screenshot — not saved       │
│  7. Update in-RAM working memory         │
│  8. Repeat                               │
└──────────────────────────────────────────┘
       ↓
Final screenshot saved to logs/ on completion
```

---

## Memory System

| Type | Where Stored | Lifetime | Purpose |
|---|---|---|---|
| **Credential memory** | `memory/user_memory.json` | Permanent | Your logins, addresses, PII |
| **Working memory** | RAM only (Python dict) | Task duration only | Agent tracks its progress, state, blockers |
| **Screenshots** | RAM only (base64) | Seconds — discarded after LLM reads them | Visual input for LLM decision-making |
| **Final screenshot** | `logs/final_*.png` | Until you delete | Proof of task completion |

**Working memory is never written to disk.** It is a JSON dict held in RAM that the agent updates each round to track what it has done, its current state, and any blockers. It is passed to the LLM each round as context so the agent remembers what it already tried.

---

## Setup (Run Once)

Follow all four steps before launching.

---

### Step 1 — Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com).

- **macOS / Linux**: Run the installer, then start Ollama:
  ```bash
  ollama serve
  ```
- **Windows**: Run the `.exe` installer. Ollama starts automatically in the system tray.

Verify Ollama is running — visit [http://localhost:11434](http://localhost:11434) and you should see `Ollama is running`.

---

### Step 2 — Pull a Vision-Capable Model

This version **requires a model that supports image input**. Pull at least one:

```bash
ollama pull llama3.2-vision
```

**Recommended vision models:**

| Model | Command | Notes |
|---|---|---|
| llama3.2-vision ✅ | `ollama pull llama3.2-vision` | Best overall (recommended) |
| llava:13b | `ollama pull llava:13b` | Higher accuracy, slower |
| llava | `ollama pull llava` | Lighter, faster |
| minicpm-v | `ollama pull minicpm-v` | Very fast, lower accuracy |

> ⚠️ Text-only models (llama3.2, mistral, etc.) will **not work** with v2 — they cannot process screenshots.

Confirm the model downloaded:
```bash
ollama list
```

---

### Step 3 — Install Python Dependencies & Chromium

From the `browser-agent/` folder:

```bash
python setup.py
```

This installs all packages from `requirements.txt` and downloads Playwright's Chromium browser. Run once only.

---

### Step 4 — Run the App

```bash
python app.py
```

Open your browser:
```
http://localhost:5000
```

---

## Adding Credentials

Open the **Memory / Credentials** section in the left sidebar and add key/value pairs. The agent substitutes `{{key}}` placeholders with real values at runtime.

**Example keys:**

| Key | Example Value |
|---|---|
| `linkedin_email` | `you@email.com` |
| `linkedin_password` | `yourpassword` |
| `walmart_email` | `you@email.com` |
| `walmart_password` | `yourpassword` |
| `delivery_address` | `123 Main St, Mississauga, ON` |
| `full_name` | `Your Name` |

Keys containing `password`, `pin`, `secret`, `card`, or `token` are **blurred in the UI**. Hover to reveal. All data lives only in `memory/user_memory.json` — never sent anywhere.

---

## Example Tasks

- *"Login to LinkedIn for me"*
- *"Go to walmart.ca and reorder my last grocery order"*
- *"Open Gmail and summarize the 3 most recent unread emails"*
- *"Search Amazon Canada for mechanical keyboards under $100 and show me the top result"*
- *"Find the top post on reddit.com/r/programming today"*

---

## UI Panels

| Panel | What it shows |
|---|---|
| **Agent Log** | Real-time step-by-step actions |
| **Vision Feed** | What the LLM sees and thinks each round (observation + reasoning + working memory) |
| **Screenshots** | Final screenshots only (intermediate ones are discarded) |
| **Errors** | Failed actions with reason |

---

## Project Structure

```
browser-agent/
├── app.py                  ← Flask + SocketIO server
├── agent.py                ← Vision LLM planner + Playwright executor
├── setup.py                ← One-time installer
├── requirements.txt
├── README.md
├── templates/
│   └── index.html          ← Web UI
├── memory/
│   └── user_memory.json    ← Your credentials (local only)
└── logs/
    ├── agent.log
    ├── errors.log
    └── final_*.png         ← Final screenshots only
```

---

## Tips

- **Max Rounds** — increase to 25+ for complex multi-page tasks
- **Headless mode** — faster, but you can't see what's happening
- **CAPTCHA / MFA** — the agent will report this as a blocker in the Vision Feed; you may need to handle it manually
- **Model accuracy** — if the agent misreads screens, try `llava:13b` for better visual understanding
