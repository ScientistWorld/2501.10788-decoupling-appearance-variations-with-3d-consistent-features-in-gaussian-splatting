#!/usr/bin/env python3
"""Run vanilla 3DGS, DAVIGS and its ablations/variants on the benchmark scenes.

Each run writes ``<out>/predictions.pt`` with
  {"novel_view": {scene_id: [6, 3, H, W // 2]},           # held-out right halves, in [0, 1]
   "scene": {scene_id: {"means", "scales", "quats", "opacity", "sh"}},   # the 3D Gaussians
   "average_appearance": {scene_id: [6, 3, H, W]}}       # own appearance-free renders (not scored)
The evaluator renders "scene" at the canonical cameras itself. Per-scene
results are cached under ``<out>/scenes/`` so interrupted
jobs resume. Tasks are spread over the job's GPUs (several workers per GPU).

Two appearance-module hyperparameters are not given in the paper: the hash
grid's finest resolution and the module learning rates. ``--plan select`` runs
the candidates below on the development scenes; ``method/select_config.py``
scores them with the public development-scene scorer and writes
``method/selected_config.json``, which every other plan reads.

Examples
  python -m method.run_experiment --plan select
  python -m method.run_experiment --plan reference,audit
  python -m method.run_experiment --plan proposed --steps 600
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import torch

from data.glav_data import scene_ids

HERE = Path(__file__).resolve().parent
SELECTED = HERE / "selected_config.json"

METHODS = {
    "vanilla_3dgs": {"kind": "vanilla"},
    "davigs": {"kind": "davigs", "feature_mode": "xyz", "use_id": True},
    "davigs_uv_depth": {"kind": "davigs", "feature_mode": "uv_depth", "use_id": True},
    "davigs_no_id": {"kind": "davigs", "feature_mode": "xyz", "use_id": False},
}
DEFAULTS = {
    "steps": 6000,
    "max_gaussians": 6000,
    "feature_mode": "xyz",
    "use_id": True,
    # Paper: 8 px cells at 1080p (0.74% of the image height). At 96 px that is
    # below one pixel, so every pixel is queried (paper Table 5: cell 1 == cell 8).
    "cell_size": 1,
    "emb_dim": 32,
    "hash_max_res": 512,
    "lr_emb": 5e-3,
    "lr_mlp": 1e-3,
    "lr_grid": 5e-3,
    "module_lr_scale": 1.0,
    "module_cosine": False,
    "post_adapt_steps": 0,
    "heldout_adapt": True,
}
SELECT_KEYS = ("hash_max_res", "module_lr_scale")
CANDIDATES = {
    "r512_lr1": {"hash_max_res": 512, "module_lr_scale": 1.0},
    "r128_lr1": {"hash_max_res": 128, "module_lr_scale": 1.0},
    "r512_lr02": {"hash_max_res": 512, "module_lr_scale": 0.2},
    "r128_lr02": {"hash_max_res": 128, "module_lr_scale": 0.2},
}
VARIANTS = {
    "reference": {},
    "module_cosine": {"module_cosine": True, "post_adapt_steps": 200},
    "long": {"steps_mult": 1.5},
    "short": {"steps_mult": 0.25},
    "noadapt": {"heldout_adapt": False},
    **{f"cand_{k}": dict(v) for k, v in CANDIDATES.items()},
}
AUDIT = [
    ("long", [0]),
    ("reference", [1, 2]),
    ("module_cosine", [0, 1, 2]),
    ("noadapt", [0]),
    ("short", [0]),
]


def selected() -> dict:
    if SELECTED.exists():
        sel = json.loads(SELECTED.read_text())
        return {k: sel[k] for k in SELECT_KEYS}
    return {}


def plan(name: str) -> list[dict]:
    if name == "select":
        return [dict(out=f"outputs/_select/{c}/davigs", method="davigs", variant=f"cand_{c}", seed=0, scenes="dev") for c in CANDIDATES]
    if name == "reference":
        return [dict(out=f"outputs/{m}", method=m, variant="reference", seed=0, scenes="all") for m in METHODS]
    if name == "baseline":
        return [dict(out="outputs/vanilla_3dgs", method="vanilla_3dgs", variant="reference", seed=0, scenes="all")]
    if name == "proposed":
        return [dict(out="outputs/davigs", method="davigs", variant="reference", seed=0, scenes="all")]
    if name == "ablations":
        return [dict(out=f"outputs/{m}", method=m, variant="reference", seed=0, scenes="all") for m in ("davigs_uv_depth", "davigs_no_id")]
    if name == "audit":
        return [
            dict(out=f"outputs/_audit/{v}_s{s}/davigs", method="davigs", variant=v, seed=s, scenes="test")
            for v, seeds in AUDIT
            for s in seeds
        ]
    raise SystemExit(f"unknown plan {name}")


def resolve(method: str, variant: str, steps_override: int | None) -> dict:
    cfg = dict(DEFAULTS)
    if not variant.startswith("cand_"):
        cfg.update(selected())
    cfg.update(METHODS[method])
    var = dict(VARIANTS[variant])
    mult = var.pop("steps_mult", 1.0)
    cfg.update(var)
    if steps_override:
        cfg["steps"] = int(steps_override)
    cfg["steps"] = max(10, int(round(cfg["steps"] * mult)))
    return cfg


def code_version() -> str:
    h = hashlib.sha1()
    for f in sorted(HERE.glob("*.py")):
        if f.name != "select_config.py":
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def cfg_hash(cfg: dict, seed: int) -> str:
    """Cache key: configuration, seed and the method source code."""
    key = {"cfg": cfg, "seed": seed, "code": code_version()}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def pick_devices() -> list[str]:
    """GPUs for this job: at most GLAV_MAX_GPUS (default 2, the job's request),
    preferring the devices with the most free memory."""
    if not torch.cuda.is_available():
        return []
    free = []
    for i in range(torch.cuda.device_count()):
        f, total = torch.cuda.mem_get_info(i)
        print(f"[run] cuda:{i} free {f / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
        if f >= 0.6 * total:
            free.append((f, i))
    limit = int(os.environ.get("GLAV_MAX_GPUS", "2"))
    return [f"cuda:{i}" for _, i in sorted(free, reverse=True)[:limit]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="reference", help="plan name or comma-separated plans (run in priority order)")
    ap.add_argument("--steps", type=int, default=None, help="override optimisation steps (smoke tests)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any run is incomplete")
    args = ap.parse_args()

    runs = [r for name in args.plan.split(",") for r in plan(name.strip())]
    tasks = []
    for run in runs:
        cfg = resolve(run["method"], run["variant"], args.steps)
        run["cfg"] = cfg
        run["hash"] = cfg_hash(cfg, run["seed"])
        run["sids"] = scene_ids(run["scenes"])
        for sid in run["sids"]:
            cache = Path(run["out"]) / "scenes" / f"{sid}-{run['hash']}.pt"
            if not cache.exists():
                label = f"{run['method']}/{run['variant']}/s{run['seed']}"
                tasks.append(dict(cache=str(cache), cfg=cfg, seed=run["seed"], sid=sid, label=label))

    ncpu = len(os.sched_getaffinity(0))
    devices = pick_devices()
    ngpu = len(devices)
    if ngpu:
        per_gpu = int(os.environ.get("GLAV_WORKERS_PER_GPU", "6"))
        n_workers = args.workers or max(1, min(len(tasks), per_gpu * ngpu, max(ncpu, ngpu)))
        threads = 1
    else:
        devices = ["cpu"]
        n_workers = args.workers or max(1, min(len(tasks), ncpu))
        threads = max(1, ncpu // max(1, n_workers))
    print(f"[run] plan={args.plan} runs={len(runs)} pending_tasks={len(tasks)} gpus={ngpu} cpus={ncpu} workers={n_workers}", flush=True)
    for k in SELECT_KEYS:
        print(f"[run] selected {k} = {selected().get(k, DEFAULTS[k])}", flush=True)

    if tasks:
        from method.worker import worker

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for t in tasks:  # FIFO: earlier plans first
            queue.put(t)
        for _ in range(n_workers):
            queue.put(None)
        procs = [ctx.Process(target=worker, args=(i, devices[i % len(devices)], threads, queue)) for i in range(n_workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    incomplete = []
    for run in runs:
        out = Path(run["out"])
        caches = {sid: out / "scenes" / f"{sid}-{run['hash']}.pt" for sid in run["sids"]}
        missing = [sid for sid, c in caches.items() if not c.exists()]
        if missing:
            incomplete.append((run["out"], missing))
            continue
        preds = {"novel_view": {}, "scene": {}, "average_appearance": {}}
        logs = {}
        for sid, c in caches.items():
            r = torch.load(c, map_location="cpu", weights_only=True)
            preds["novel_view"][sid] = r["novel_view"]
            preds["scene"][sid] = r["scene"]
            preds["average_appearance"][sid] = r["average_appearance"]
            logs[sid] = r["log"]
        torch.save(preds, out / "predictions.pt")
        (out / "run_info.json").write_text(
            json.dumps({"method": run["method"], "variant": run["variant"], "seed": run["seed"], "config": run["cfg"], "scenes": logs}, indent=2)
            + "\n"
        )
        print(f"[run] wrote {out / 'predictions.pt'}", flush=True)
    for out, missing in incomplete:
        print(f"[run] INCOMPLETE {out}: missing {missing}", flush=True)
    if incomplete and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
