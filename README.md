# BrowserAgent

A local AI browser automation agent. Uses a locally running Ollama LLM to control a Chromium browser via natural language instructions.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- At least one LLM pulled: `ollama pull llama3.2`

## Setup (run once)

```bash
cd browser-agent
python setup.py        # installs pip packages + Chromium
python app.py          # starts the server
# Open http://localhost:5000
```

## How it works

1. You type a task in the UI (e.g. "Go to Walmart and reorder my last grocery order")
2. The app sends the task + your stored memory (credentials, preferences) to Ollama
3. Ollama returns a JSON plan of browser actions
4. Playwright executes each action on a real Chromium browser window
5. The agent re-plans after each round using the live page state
6. All actions and errors are logged in `logs/`

## Memory

Store credentials and personal info in the **Memory** panel in the UI, or directly in `memory/user_memory.json`.

**Example:**
```json
{
  "walmart_email":    "you@email.com",
  "walmart_password": "yourpassword",
  "delivery_address": "123 Main St, Mississauga, ON",
  "full_name":        "Your Name"
}
```

> Sensitive keys (password, pin, card, etc.) are blurred in the UI — hover to reveal.

## Logs

- `logs/agent.log`  — all actions
- `logs/errors.log` — errors only
- `logs/*.png`      — screenshots taken during tasks

## Supported Actions

| Action | Description |
|---|---|
| navigate | Go to a URL |
| click | Click a button or link |
| fill | Fill an input field |
| type | Type text character by character |
| press | Press a key (Enter, Tab, Escape) |
| scroll | Scroll the page |
| wait | Pause for N milliseconds |
| screenshot | Take a screenshot |
| get_text | Read text from an element |
| select_option | Choose a dropdown option |
| done | Mark task complete |
| error | Log an error and stop |

## Recommended Models

| Model | Command | Notes |
|---|---|---|
| llama3.2 | `ollama pull llama3.2` | Best overall, fast |
| llama3.1 | `ollama pull llama3.1` | Larger, more capable |
| mistral | `ollama pull mistral` | Good JSON reliability |
| qwen2.5 | `ollama pull qwen2.5` | Strong structured output |
