"""Worker process: trains one (configuration, seed, scene) task at a time."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import torch


def run_task(task: dict, device: str, tag: str) -> None:
    from data.glav_data import load_scene
    from method.train_scene import train_scene

    cache = Path(task["cache"])
    if cache.exists():
        return
    scene = load_scene(task["sid"])

    def log(msg: str) -> None:
        print(f"[{tag}] {task['label']} {task['sid']}: {msg}", flush=True)

    result = train_scene(scene, task["cfg"], int(task["seed"]), device, log=log)
    result["config"] = task["cfg"]
    result["seed"] = int(task["seed"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save(result, tmp)
    os.replace(tmp, cache)


def worker(worker_id: int, device: str, threads: int, queue) -> None:
    torch.set_num_threads(max(1, threads))
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    tag = f"w{worker_id}:{device}"
    while True:
        task = queue.get()
        if task is None:
            break
        t0 = time.time()
        try:
            run_task(task, device, tag)
            print(f"[{tag}] done {task['label']} {task['sid']} in {time.time() - t0:.0f}s", flush=True)
        except Exception:  # keep going; the assembler reports missing scenes
            print(f"[{tag}] FAILED {task['label']} {task['sid']}\n{traceback.format_exc()}", flush=True)
