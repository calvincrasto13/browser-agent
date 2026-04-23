"""
BrowserAgent Setup Script — run once to install all dependencies.
"""
import subprocess, sys, json, pathlib

def run(cmd, desc):
    print(f"\n{'='*52}\n  {desc}\n{'='*52}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARN] Exited with code {result.returncode}")
    return result.returncode == 0

print("\n" + "="*52)
print("  BrowserAgent — First-Time Setup")
print("="*52)

run(f"{sys.executable} -m pip install -r requirements.txt",
    "Installing Python packages")

run(f"{sys.executable} -m playwright install chromium",
    "Installing Chromium browser")

pathlib.Path("logs").mkdir(exist_ok=True)
pathlib.Path("memory").mkdir(exist_ok=True)

mem_path = pathlib.Path("memory/user_memory.json")
if not mem_path.exists():
    example = pathlib.Path("memory/user_memory.example.json")
    if example.exists():
        import shutil
        shutil.copy(example, mem_path)
    else:
        mem_path.write_text(json.dumps(
            {"_note": "Add your credentials here via the UI or this file"},
            indent=2
        ))
    print("[OK] Created memory/user_memory.json — add your credentials")

print("\n" + "="*52)
print("  Setup complete!")
print("  Run:  python app.py")
print("  Then: http://localhost:5000")
print("="*52 + "\n")
