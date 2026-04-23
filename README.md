# BrowserAgent

Local AI browser automation agent — uses **Ollama** (local LLM) to control a Chromium tab via natural language instructions.

---

## Setup

### 1. Install Ollama

Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

- **Windows/Mac**: Run the installer — Ollama starts automatically in the background.
- **Linux**: Run `curl -fsSL https://ollama.com/install.sh | sh` then start it with `ollama serve`.

Verify it is running:
```bash
ollama list
```

---

### 2. Pull a Model

BrowserAgent works best with instruction-following models. Pull one before running the app:

```bash
ollama pull llama3.2
```

Other recommended options:

| Model | Pull Command | Notes |
|---|---|---|
| **llama3.2** (default) | `ollama pull llama3.2` | Best speed/quality balance |
| **llama3.1** | `ollama pull llama3.1` | Better for complex multi-step tasks |
| **mistral** | `ollama pull mistral` | Reliable JSON output |
| **qwen2.5** | `ollama pull qwen2.5` | Strong structured output |

Verify your model downloaded:
```bash
ollama list
```

---

### 3. Install Python Dependencies & Chromium

```bash
python setup.py
```

This installs all pip packages and downloads the Playwright Chromium browser automatically.

---

### 4. Run the App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Storing Login Credentials

BrowserAgent uses the Memory system to securely store your usernames, passwords, and other personal info — all saved locally on your PC in `memory/user_memory.json`. The LLM reads these values automatically when it needs to log in to a site.

### How to Add Credentials via the UI

1. Open **http://localhost:5000**
2. In the left sidebar, go to the **Memory** tab
3. Fill in the **Key** and **Value** fields and click **+ Add**

Use a consistent naming pattern so the LLM can recognize them:

| Key | Value (your actual value) |
|---|---|
| `walmart_email` | `you@email.com` |
| `walmart_password` | `yourpassword123` |
| `amazon_email` | `you@email.com` |
| `amazon_password` | `yourpassword123` |
| `gmail_email` | `you@gmail.com` |
| `gmail_password` | `yourpassword` |
| `full_name` | `Jane Smith` |
| `delivery_address` | `123 Main St, Mississauga, ON L5B 2C9` |
| `phone_number` | `647-555-0123` |

> **Tip:** Name keys as `<sitename>_email` and `<sitename>_password`. The LLM will match them to the right site automatically when you give it a task.

---

### How to Add Credentials Directly (JSON file)

You can also edit `memory/user_memory.json` directly in any text editor:

```json
{
  "walmart_email":       "you@email.com",
  "walmart_password":    "yourpassword",
  "amazon_email":        "you@email.com",
  "amazon_password":     "yourpassword",
  "netflix_email":       "you@email.com",
  "netflix_password":    "yourpassword",
  "full_name":           "Jane Smith",
  "delivery_address":    "123 Main St, Mississauga, ON L5B 2C9",
  "phone_number":        "647-555-0123"
}
```

Save the file — changes take effect on the next task run (no restart needed).

---

### Security Notes

- All credentials stay **100% local** — they are never sent anywhere except to your local Ollama instance running on the same machine.
- In the UI, any key containing `password`, `pin`, `card`, `cvv`, `secret`, or `token` is **blurred** — hover over it to reveal the value.
- Do **not** commit `memory/user_memory.json` to Git or share it. Add it to `.gitignore`:
  ```
  memory/user_memory.json
  logs/
  ```

---

### Example Tasks Using Stored Credentials

Once your credentials are in memory, you can give natural language tasks like:

```
Log into my Walmart account and reorder my last grocery order
```

```
Sign into Amazon and check my recent orders
```

```
Go to Netflix and resume the show I was watching
```

The agent will automatically look up your stored email and password for that site, fill in the login form, and proceed with the task.

---

### Two-Factor Authentication (2FA)

If a site uses 2FA (a code sent to your phone or email), the agent will **pause at that step** — it cannot receive the code automatically. When this happens:

1. Watch the browser window that opens on your screen
2. Manually type the 2FA code into the browser when prompted
3. The agent will detect the page has moved on and continue automatically

For sites you use frequently with 2FA, consider using an **app password** (Gmail, Outlook) or disabling 2FA on a trusted device to allow unattended automation.
