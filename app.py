"""
Flask + SocketIO server for BrowserAgent.
Run: python app.py  →  http://localhost:5000
"""
import asyncio, json, threading
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from agent import run_task, load_memory, save_memory, LOGS_DIR, warmup_model, GPU_CFG

app      = Flask(__name__)
app.secret_key = "browser-agent-secret-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

ACTIVE_TASKS: dict = {}


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Memory API ────────────────────────────────────────────────────────────────
@app.route("/api/memory", methods=["GET"])
def get_memory():
    return jsonify(load_memory())

@app.route("/api/memory", methods=["POST"])
def set_memory():
    save_memory(request.json or {})
    return jsonify({"status": "saved"})

@app.route("/api/memory/entry", methods=["POST"])
def add_memory_entry():
    data = request.json or {}
    key  = data.get("key", "").strip()
    val  = data.get("value", "").strip()
    if not key:
        return jsonify({"error": "key required"}), 400
    mem = load_memory()
    mem[key] = val
    save_memory(mem)
    return jsonify({"status": "ok", "memory": mem})

@app.route("/api/memory/entry/<key>", methods=["DELETE"])
def del_memory_entry(key):
    mem = load_memory()
    mem.pop(key, None)
    save_memory(mem)
    return jsonify({"status": "deleted", "memory": mem})


# ── Logs API ──────────────────────────────────────────────────────────────────
@app.route("/api/logs")
def get_logs():
    log_file = LOGS_DIR / "agent.log"
    err_file = LOGS_DIR / "errors.log"
    agent_lines = log_file.read_text().splitlines()[-100:] if log_file.exists() else []
    error_lines = err_file.read_text().splitlines()[-50:]  if err_file.exists() else []
    return jsonify({"agent": agent_lines, "errors": error_lines})

@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    for f in LOGS_DIR.glob("*.log"):
        f.write_text("")
    return jsonify({"status": "cleared"})

@app.route("/api/screenshots")
def list_screenshots():
    shots = sorted(LOGS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([p.name for p in shots[:20]])

@app.route("/api/screenshots/<name>")
def get_screenshot(name):
    return send_from_directory(str(LOGS_DIR), name)


# ── Ollama models ─────────────────────────────────────────────────────────────
@app.route("/api/models")
def list_models():
    try:
        import ollama
        resp   = ollama.list()
        models = [m["name"] for m in resp.get("models", [])]
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


# ── GPU status API ────────────────────────────────────────────────────────────
@app.route("/api/gpu")
def gpu_status():
    info = dict(GPU_CFG)  # base config from setup
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            info["vram_used_gb"]  = round(g.memoryUsed  / 1024, 2)
            info["vram_total_gb"] = round(g.memoryTotal / 1024, 2)
            info["gpu_load_pct"]  = round(g.load * 100, 1)
            info["gpu_temp_c"]    = g.temperature
    except Exception:
        pass
    return jsonify(info)


# ── Warm-up endpoint ──────────────────────────────────────────────────────────
@app.route("/api/warmup", methods=["POST"])
def warmup():
    model = (request.json or {}).get("model", "llama3.2-vision")
    threading.Thread(target=warmup_model, args=(model,), daemon=True).start()
    return jsonify({"status": "warming up", "model": model})


# ── SocketIO ──────────────────────────────────────────────────────────────────
@socketio.on("run_task")
def handle_run_task(data):
    sid   = request.sid
    task  = data.get("task", "").strip()
    model = data.get("model", "llama3.2-vision")

    if not task:
        emit("error", {"msg": "No task provided."})
        return

    emit("log", {"msg": f"Task started: {task}",
                 "ts": datetime.now().strftime("%H:%M:%S")})

    def _emit(event, payload):
        socketio.emit(event, payload, to=sid)

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                run_task(task, model=model, headless=False, emit_fn=_emit)
            )
            _emit("task_complete", {
                "result":     result["result"],
                "screenshot": result.get("screenshot"),
                "elapsed":    result.get("elapsed"),
            })
        except Exception as e:
            _emit("task_error", {"msg": str(e)})
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    ACTIVE_TASKS[sid] = {"thread": t, "task": task}
    t.start()


@socketio.on("disconnect")
def handle_disconnect():
    ACTIVE_TASKS.pop(request.sid, None)


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  BrowserAgent v3 — GPU + Comet-style control")
    gpu_label = GPU_CFG.get("gpu_name", "CPU")
    layers    = GPU_CFG.get("num_gpu_layers", 0)
    print(f"  GPU: {gpu_label}  |  Layers offloaded: {layers}")
    print("  Open http://localhost:5000")
    print("="*52 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
