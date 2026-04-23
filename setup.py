"""
BrowserAgent Setup Script — run once to install all dependencies.
Auto-detects GPU and prints recommended Ollama settings.
"""
import subprocess, sys, json, pathlib, platform

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
    mem_path.write_text(json.dumps(
        {"_note": "Add your credentials here via the UI or this file"},
        indent=2
    ))
    print("[OK] Created memory/user_memory.json")

# ── GPU Detection ─────────────────────────────────────────────────────────────
print("\n" + "="*52)
print("  GPU Detection")
print("="*52)

vram_gb   = 0
gpu_name  = "None detected"
gpu_type  = "cpu"

# NVIDIA via GPUtil
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        g        = gpus[0]
        vram_gb  = round(g.memoryTotal / 1024, 1)
        gpu_name = g.name
        gpu_type = "nvidia"
        print(f"[GPU] NVIDIA detected: {gpu_name}  VRAM: {vram_gb} GB")
except Exception:
    pass

# Apple Silicon
if gpu_type == "cpu" and platform.system() == "Darwin" and platform.processor() == "arm":
    try:
        import subprocess as sp
        out = sp.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        total_gb = int(out) / (1024**3)
        # Apple unified memory — GPU shares system RAM; use ~75%
        vram_gb  = round(total_gb * 0.75, 1)
        gpu_name = f"Apple Silicon (unified {round(total_gb,0):.0f} GB)"
        gpu_type = "apple"
        print(f"[GPU] Apple Silicon detected. Unified RAM: {round(total_gb,1)} GB  Usable: {vram_gb} GB")
    except Exception:
        pass

# AMD via rocm-smi
if gpu_type == "cpu":
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if "Total Memory" in line:
                mb = int(line.split()[-1])
                vram_gb  = round(mb / 1024, 1)
                gpu_name = "AMD GPU (ROCm)"
                gpu_type = "amd"
                print(f"[GPU] AMD ROCm detected. VRAM: {vram_gb} GB")
                break
    except Exception:
        pass

if gpu_type == "cpu":
    print("[GPU] No GPU detected — will run on CPU (slower)")

# Recommend num_gpu layers for Ollama
if vram_gb >= 24:
    recommended_layers = 99
elif vram_gb >= 16:
    recommended_layers = 60
elif vram_gb >= 12:
    recommended_layers = 45
elif vram_gb >= 8:
    recommended_layers = 30
elif vram_gb >= 4:
    recommended_layers = 15
else:
    recommended_layers = 0  # CPU

# Write gpu_config.json for agent.py to read at runtime
gpu_cfg = {
    "gpu_type":          gpu_type,
    "gpu_name":          gpu_name,
    "vram_gb":           vram_gb,
    "num_gpu_layers":    recommended_layers,
    "use_mmap":          True,
    "use_mlock":         vram_gb >= 8,
    "num_thread":        None,  # let Ollama auto-detect
}
pathlib.Path("gpu_config.json").write_text(json.dumps(gpu_cfg, indent=2))

print(f"\n[GPU CONFIG]")
print(f"  GPU:              {gpu_name}")
print(f"  VRAM:             {vram_gb} GB")
print(f"  num_gpu_layers:   {recommended_layers}  (written to gpu_config.json)")
print(f"  use_mlock:        {gpu_cfg['use_mlock']}")

print("\n" + "="*52)
print("  Setup complete!")
print("  Run:  python app.py")
print("  Then: http://localhost:5000")
print("="*52 + "\n")
