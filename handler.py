"""RunPod serverless handler: train a Chroma LoRA of one person from a handful of
pictures with ostris/ai-toolkit, save it on the network volume, report progress.

input: {
  name: "lily",                       # file name for the LoRA (letters, digits, - _)
  trigger: "lilyc4f5",                # word that summons her in prompts
  images: [{image: <base64>, caption: "lilyc4f5, photo of a woman ..."}, ...],
  steps: 2000, rank: 16, resolution: 1024   # optional
}
output: {lora: "lily.safetensors", trigger, steps, seconds}
"""
import base64, os, re, shutil, subprocess, time, json
import runpod, yaml

VOL = "/runpod-volume"
TOOLKIT = next((p for p in ["/app/ai-toolkit", "/ai-toolkit", "/workspace/ai-toolkit", "/opt/ai-toolkit"] if os.path.exists(os.path.join(p, "run.py"))), None)


def handler(job):
    inp = job.get("input") or {}
    if TOOLKIT is None:
        return {"error": "ai-toolkit not found in this image"}
    name = re.sub(r"[^a-z0-9_-]", "", str(inp.get("name", "her")).lower())[:40] or "her"
    trigger = re.sub(r"[^a-z0-9]", "", str(inp.get("trigger", "")).lower()) or f"{name}xz"
    images = inp.get("images") or []
    if len(images) < 4:
        return {"error": f"need at least 4 pictures, got {len(images)}"}
    steps = max(200, min(int(inp.get("steps") or 2000), 6000))
    rank = max(4, min(int(inp.get("rank") or 16), 64))
    resolution = int(inp.get("resolution") or 1024)
    base = inp.get("base") or f"{VOL}/models/unet/Chroma1-HD.safetensors"
    if not os.path.exists(base):
        base = "lodestones/Chroma1-HD"  # downloaded into the volume's HF cache on first use

    ds = f"/tmp/ds_{name}"
    out = f"/tmp/out_{name}"
    shutil.rmtree(ds, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(ds)
    for i, im in enumerate(images):
        data = base64.b64decode(str(im.get("image", "")).split(",")[-1])
        if len(data) < 50:
            continue
        ext = "png" if data[:4] == b"\x89PNG" else "jpg"
        with open(f"{ds}/{i:03d}.{ext}", "wb") as f:
            f.write(data)
        with open(f"{ds}/{i:03d}.txt", "w") as f:
            f.write(str(im.get("caption") or f"{trigger}, photo of a woman"))

    cfg = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": out,
                    "device": "cuda:0",
                    "trigger_word": trigger,
                    "network": {"type": "lora", "linear": rank, "linear_alpha": rank},
                    "save": {"dtype": "float16", "save_every": 500, "max_step_saves_to_keep": 2},
                    "datasets": [
                        {
                            "folder_path": ds,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [512, 768, resolution],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": 1e-4,
                        "dtype": "bf16",
                        "ema_config": {"use_ema": True, "ema_decay": 0.99},
                    },
                    "model": {"name_or_path": base, "arch": "chroma", "quantize": True},
                    "sample": {"sampler": "flowmatch", "sample_every": 1000000, "width": 1024, "height": 1024, "prompts": [], "guidance_scale": 4, "sample_steps": 25},
                }
            ],
        },
        "meta": {"name": name, "version": "1.0"},
    }
    cfg_path = f"/tmp/{name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    started = time.time()
    env = dict(os.environ)
    env.setdefault("HF_HOME", f"{VOL}/hf-cache")
    proc = subprocess.Popen(["python", "run.py", cfg_path], cwd=TOOLKIT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    last_report = 0
    tail = []
    buf = ""
    while True:
        ch = proc.stdout.read(1)
        if ch == "" and proc.poll() is not None:
            break
        if ch in ("\r", "\n"):
            line = buf.strip()
            buf = ""
            if line:
                tail = (tail + [line])[-40:]
                m = re.search(r"(\d+)/(\d+)\s*\[", line)
                if m and time.time() - last_report > 15:
                    last_report = time.time()
                    step, total = int(m.group(1)), int(m.group(2))
                    runpod.serverless.progress_update(job, json.dumps({"step": step, "total": total, "seconds": int(time.time() - started)}))
        else:
            buf += ch
    if proc.returncode != 0:
        return {"error": "training failed", "log": "\n".join(tail[-15:])}

    final = os.path.join(out, name, f"{name}.safetensors")
    if not os.path.exists(final):
        cands = sorted([p for p in os.listdir(os.path.join(out, name)) if p.endswith(".safetensors")]) if os.path.isdir(os.path.join(out, name)) else []
        if not cands:
            return {"error": "training produced no file", "log": "\n".join(tail[-15:])}
        final = os.path.join(out, name, cands[-1])
    dest_dir = f"{VOL}/models/loras"
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{name}.safetensors")
    shutil.copyfile(final, dest)
    shutil.rmtree(ds, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    return {"lora": f"{name}.safetensors", "trigger": trigger, "steps": steps, "seconds": int(time.time() - started)}


runpod.serverless.start({"handler": handler})
