#!/usr/bin/env python3
"""Run vanilla 3DGS, DAVIGS and its ablations/variants on the benchmark scenes.

Each run writes ``<out>/predictions.pt`` with
  {"novel_view": {scene_id: [V, 3, H, W // 2]},           # held-out right halves
   "average_appearance": {scene_id: [V, 3, H, W]}}       # appearance-free renders
in [0, 1]; per-scene results are cached under ``<out>/scenes/`` so interrupted
jobs resume. Tasks are spread over all visible GPUs (several workers per GPU).

Examples
  python -m method.run_experiment --plan reference
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
    "cell_size": 2,
    "emb_dim": 32,
    "lr_emb": 5e-3,
    "lr_mlp": 1e-3,
    "lr_grid": 5e-3,
    "module_cosine": False,
    "post_adapt_steps": 0,
    "heldout_adapt": True,
}
VARIANTS = {
    "reference": {},
    "module_cosine": {"module_cosine": True, "post_adapt_steps": 200},
    "cell1": {"cell_size": 1},
    "long": {"steps_mult": 1.5},
    "short": {"steps_mult": 0.25},
    "noadapt": {"heldout_adapt": False},
}
AUDIT = [
    ("reference", [1, 2]),
    ("module_cosine", [0, 1, 2]),
    ("cell1", [0, 1, 2]),
    ("long", [0, 1, 2]),
    ("short", [0, 1, 2]),
    ("noadapt", [0]),
]


def plan(name: str) -> list[dict]:
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
    cfg.update(METHODS[method])
    var = dict(VARIANTS[variant])
    mult = var.pop("steps_mult", 1.0)
    cfg.update(var)
    if steps_override:
        cfg["steps"] = int(steps_override)
    cfg["steps"] = max(10, int(round(cfg["steps"] * mult)))
    return cfg


def cfg_hash(cfg: dict, seed: int) -> str:
    return hashlib.sha1(json.dumps({"cfg": cfg, "seed": seed}, sort_keys=True).encode()).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="reference")
    ap.add_argument("--steps", type=int, default=None, help="override optimisation steps (smoke tests)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any run is incomplete")
    args = ap.parse_args()

    runs = plan(args.plan)
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

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    ncpu = len(os.sched_getaffinity(0))
    if ngpu:
        devices = [f"cuda:{i}" for i in range(ngpu)]
        per_gpu = int(os.environ.get("GLAV_WORKERS_PER_GPU", "4"))
        n_workers = args.workers or max(1, min(len(tasks), per_gpu * ngpu, max(ncpu, ngpu)))
        threads = 1
    else:
        devices = ["cpu"]
        n_workers = args.workers or max(1, min(len(tasks), ncpu))
        threads = max(1, ncpu // max(1, n_workers))
    print(f"[run] plan={args.plan} runs={len(runs)} pending_tasks={len(tasks)} gpus={ngpu} cpus={ncpu} workers={n_workers}", flush=True)

    if tasks:
        from method.worker import worker

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for t in tasks:
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
        preds = {"novel_view": {}, "average_appearance": {}}
        logs = {}
        for sid, c in caches.items():
            r = torch.load(c, map_location="cpu", weights_only=True)
            preds["novel_view"][sid] = r["novel_view"]
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
