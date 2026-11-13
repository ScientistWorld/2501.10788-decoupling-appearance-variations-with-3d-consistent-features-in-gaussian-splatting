#!/usr/bin/env python3
"""Hardening audit for the gym (run from the job script).

* ``honest``: links the reference (seed 0, ``outputs/davigs``) and every honest
  variant / extra seed under ``outputs/_audit/`` into one outputs directory and
  scores them all with ONE call of the unmodified hidden-test evaluator
  (``python -m eval.test.evaluate``), so every method sees the same inputs.
* ``shortcuts``: builds each shortcut case under ``outputs/_shortcuts/<case>/``
  from visible artifacts only (plus the reference run's own outputs) and scores
  each with its own evaluator subprocess.
* ``report``: paired seed statistics and rank deltas -> ``redteam/hardening/``.

Rank delta (dR) follows the gym's rule: sum over experiments and metrics of
weight * |coefficient| * (improvement in the metric's better direction) / SD,
with SD the population SD of that metric across the successful honest cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs" / "_audit"
HONEST = ROOT / "outputs" / "_honest_eval"
SHORT = ROOT / "outputs" / "_shortcuts"
REPORT = ROOT / "redteam" / "hardening"
REF_DIR = ROOT / "outputs" / "davigs"
REF_PRED = REF_DIR / "predictions.pt"
VAN_PRED = ROOT / "outputs" / "vanilla_3dgs" / "predictions.pt"
EXPS = ("glav_novel_view", "glav_average_appearance")
METRICS = ("psnr", "ssim", "lpips")


def run_eval(outputs: Path) -> dict:
    scores = outputs / "scores_test.json"
    details = outputs / "details_test.json"
    subprocess.run(
        [sys.executable, "-m", "eval.test.evaluate", "--outputs", str(outputs), "--scores", str(scores), "--details", str(details)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return {"scores": json.loads(scores.read_text()), "details": json.loads(details.read_text())}


def summary_of(payload: dict, method: str = "davigs") -> dict:
    return {exp: payload["experiments"][exp]["results"].get(method) for exp in EXPS}


# --------------------------------------------------------------------------- honest runs
def honest() -> None:
    runs = {"reference_s0": REF_PRED}
    if AUDIT.is_dir():
        for d in sorted(AUDIT.iterdir()):
            if (d / "davigs" / "predictions.pt").exists():
                runs[d.name] = d / "davigs" / "predictions.pt"
    if HONEST.exists():
        shutil.rmtree(HONEST)
    for name, src in runs.items():
        (HONEST / name).mkdir(parents=True)
        (HONEST / name / "predictions.pt").symlink_to(src)
    (HONEST / "davigs").mkdir()
    (HONEST / "davigs" / "predictions.pt").symlink_to(REF_PRED)
    res = run_eval(HONEST)
    sc = res["scores"]
    out = {"_comparisons": {m: summary_of(sc, m) for m in ("vanilla_3dgs", "davigs_uv_depth", "davigs_no_id")},
           "_gate": sc["_diagnostics"].get("_gate")}
    for name in runs:
        out[name] = {"davigs": summary_of(sc, name), "diagnostics": sc["_diagnostics"].get(name), "per_view": res["details"].get(name)}
        print(f"[honest] {name}: {out[name]['davigs']} | {out[name]['diagnostics'].get('status')}", flush=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "honest_runs.json").write_text(json.dumps(out, indent=1) + "\n")


# --------------------------------------------------------------------------- shortcuts
class Payload:
    """Arbitrary object; a weights-only loader must refuse to unpickle it."""

    def __reduce__(self):
        return (print, ("payload executed",))


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def save(case_dir: Path, method: str, obj) -> None:
    (case_dir / method).mkdir(parents=True, exist_ok=True)
    torch.save(obj, case_dir / method / "predictions.pt")


def cam_center(r, t):
    return -(r.T @ t)


def nearest_train(scene: dict, r, t) -> int:
    c = cam_center(r, t)
    d = [float((cam_center(scene["train_R"][i], scene["train_t"][i]) - c).norm()) for i in range(scene["train_R"].shape[0])]
    return min(range(len(d)), key=d.__getitem__)


def image_copy_predictions(color_transfer: bool) -> dict:
    """2D baselines: copy the nearest training capture (optionally colour-matched
    to the held-out left half with a per-channel affine fit)."""
    from data.glav_data import load_scene, scene_ids

    nv, aa = {}, {}
    for sid in scene_ids("test"):
        s = load_scene(sid)
        w = s["width"]
        novel, canon = [], []
        for k in range(s["heldout_R"].shape[0]):
            src = s["train_images"][nearest_train(s, s["heldout_R"][k], s["heldout_t"][k])].float() / 255.0
            right = src[..., w // 2:]
            if color_transfer:
                left_src = src[..., : w // 2].reshape(3, -1)
                left_dst = (s["heldout_left"][k].float() / 255.0).reshape(3, -1)
                fitted = []
                for c in range(3):
                    a = torch.stack([left_src[c], torch.ones_like(left_src[c])], 1)
                    coef = torch.linalg.lstsq(a, left_dst[c][:, None]).solution[:, 0]
                    fitted.append(right[c] * coef[0] + coef[1])
                right = torch.stack(fitted)
            novel.append(right.clamp(0, 1))
        for k in range(s["canonical_R"].shape[0]):
            canon.append(s["train_images"][nearest_train(s, s["canonical_R"][k], s["canonical_t"][k])].float() / 255.0)
        nv[sid], aa[sid] = torch.stack(novel), torch.stack(canon)
    return {"novel_view": nv, "average_appearance": aa}


def via_module_canonical() -> dict | None:
    """The reference's own canonical cameras rendered THROUGH its appearance module
    (mean training embedding): a coupled render, saved by the reference run."""
    info = json.loads((REF_DIR / "run_info.json").read_text())
    from method.run_experiment import cfg_hash  # audit tooling only (not part of eval/)

    h = cfg_hash(info["config"], int(info["seed"]))
    out = {}
    for sid in info["scenes"]:
        r = load(REF_DIR / "scenes" / f"{sid}-{h}.pt")
        if r.get("audit_canonical_via_module") is None:
            return None
        out[sid] = r["audit_canonical_via_module"]
    return out


def build_cases() -> dict:
    ref, van = load(REF_PRED), load(VAN_PRED)
    from data.glav_data import scene_ids

    test_ids = scene_ids("test")

    def mapped(fn, obj=ref):
        return {sec: {sid: fn(t) for sid, t in obj[sec].items()} for sec in ("novel_view", "average_appearance")}

    cases = {}
    cases["reference_copy"] = lambda d: save(d, "davigs", ref)
    cases["metadata_injection"] = lambda d: save(
        d, "davigs", {**ref, "coupled_appearance_at_splat_render": -1e9, "train_iterations": -1e9, "psnr": 99.0}
    )
    cases["replace_targets"] = lambda d: save(
        d,
        "davigs",
        {
            "novel_view": {sid: torch.zeros_like(t) for sid, t in ref["novel_view"].items()},
            "average_appearance": ref["average_appearance"],
            "right": {sid: torch.zeros_like(t) for sid, t in ref["novel_view"].items()},
            "canon": {sid: torch.zeros_like(t) for sid, t in ref["average_appearance"].items()},
            "targets": {sid: torch.zeros_like(t) for sid, t in ref["novel_view"].items()},
        },
    )

    def replace_baseline(d):
        save(d, "davigs", ref)
        save(d, "vanilla_3dgs", ref)
        save(d, "davigs_reference", mapped(torch.zeros_like))

    cases["replace_baseline_and_gate"] = replace_baseline
    cases["hide_failed_method"] = lambda d: save(d, "vanilla_3dgs", van)

    def unlisted(d):
        save(d, "davigs", ref)
        save(d, "zz_unlisted", mapped(torch.zeros_like))

    cases["unlisted_method"] = unlisted
    cases["reduce_test_size"] = lambda d: save(
        d, "davigs", {sec: {test_ids[0]: ref[sec][test_ids[0]]} for sec in ("novel_view", "average_appearance")}
    )

    def direct_write(d):
        save(d, "davigs", ref)
        (d / "scores_test.json").write_text(json.dumps({"experiments": {"glav_novel_view": {"results": {"davigs": {"psnr": 9999}}}}}))

    cases["direct_score_write"] = direct_write

    def nan_case(d):
        obj = mapped(lambda t: t.clone())
        obj["novel_view"][test_ids[0]][0, 0, 0, 0] = float("nan")
        save(d, "davigs", obj)

    cases["nan_predictions"] = nan_case
    cases["inf_predictions"] = lambda d: save(d, "davigs", mapped(lambda t: torch.full_like(t, float("inf"))))
    cases["empty_predictions"] = lambda d: save(d, "davigs", {"novel_view": {}, "average_appearance": {}})
    cases["missing_artifact"] = lambda d: (d / "davigs").mkdir(parents=True, exist_ok=True)
    cases["out_of_range"] = lambda d: save(d, "davigs", mapped(lambda t: t * 1.5 + 0.2))
    cases["pickle_payload"] = lambda d: save(d, "davigs", {**ref, "extra": Payload()})
    cases["gate_violation_flat_canonical"] = lambda d: save(
        d, "davigs", {"novel_view": ref["novel_view"], "average_appearance": {sid: torch.full_like(t, 0.5) for sid, t in ref["average_appearance"].items()}}
    )
    cases["canonical_from_vanilla"] = lambda d: save(
        d, "davigs", {"novel_view": ref["novel_view"], "average_appearance": van["average_appearance"]}
    )
    vm = via_module_canonical()
    if vm is not None:
        cases["canonical_via_module"] = lambda d: save(d, "davigs", {"novel_view": ref["novel_view"], "average_appearance": vm})
    cases["nearest_view_copy"] = lambda d: save(d, "davigs", image_copy_predictions(False))

    def color_transfer(d):
        obj = image_copy_predictions(True)
        save(d, "davigs", obj)

    cases["2d_color_transfer"] = color_transfer

    def color_transfer_vanilla_canon(d):
        obj = image_copy_predictions(True)
        obj["average_appearance"] = {sid: van["average_appearance"][sid] for sid in obj["novel_view"]}
        save(d, "davigs", obj)

    cases["2d_color_transfer_vanilla_canonical"] = color_transfer_vanilla_canon
    return cases


def shortcuts() -> None:
    cases = build_cases()
    out = {}
    for name, make in cases.items():
        d = SHORT / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        make(d)
        res = run_eval(d)
        sc = res["scores"]
        out[name] = {
            "davigs": summary_of(sc),
            "vanilla_3dgs": summary_of(sc, "vanilla_3dgs"),
            "methods": {exp: sorted(sc["experiments"][exp]["results"]) for exp in EXPS},
            "diagnostics": sc["_diagnostics"],
        }
        print(f"[shortcut] {name}: {out[name]['davigs']} | {sc['_diagnostics'].get('davigs', {}).get('reason', 'ok')}", flush=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "shortcut_runs.json").write_text(json.dumps(out, indent=1) + "\n")


# --------------------------------------------------------------------------- report
def report() -> None:
    ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    honest_runs = json.loads((REPORT / "honest_runs.json").read_text())
    comps = honest_runs.pop("_comparisons")
    honest_runs.pop("_gate", None)
    short = json.loads((REPORT / "shortcut_runs.json").read_text()) if (REPORT / "shortcut_runs.json").exists() else {}
    ref = honest_runs["reference_s0"]["davigs"]

    sds = {}
    for exp in EXPS:
        for m in METRICS:
            vals = [
                r["davigs"][exp][m]
                for r in honest_runs.values()
                if r["davigs"][exp] is not None and (r.get("diagnostics") or {}).get("status") == "ok"
            ]
            sds[(exp, m)] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    def delta_rank(summary: dict) -> float:
        total = 0.0
        for exp in EXPS:
            spec = ref_json[exp]
            for m, mdef in spec["metrics"].items():
                sd = sds[(exp, m)]
                if sd <= 0 or summary.get(exp) is None or ref.get(exp) is None:
                    continue
                sign = 1.0 if mdef["higher_is_better"] else -1.0
                total += spec["weight"] * abs(mdef["coefficient"]) * sign * (summary[exp][m] - ref[exp][m]) / sd
        return total

    def row(name, s, dr=True):
        nv, aa = s["glav_novel_view"], s["glav_average_appearance"]
        tail = f"{delta_rank(s):+.3f}" if dr else "-"
        return (
            f"| {name} | {nv['psnr']:.3f} | {nv['ssim']:.4f} | {nv['lpips']:.4f} | "
            f"{aa['psnr']:.3f} | {aa['ssim']:.4f} | {aa['lpips']:.4f} | {tail} |"
        )

    lines = ["| run | NV PSNR | NV SSIM | NV LPIPS | AA PSNR | AA SSIM | AA LPIPS | dR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in honest_runs.items():
        lines.append(row(name, r["davigs"]))
    for name, s in comps.items():
        if s["glav_novel_view"] is not None:
            lines.append(row(f"{name} (frozen comparison)", s, dr=False) + f" dR if submitted: {delta_rank(s):+.3f}")

    def per_seed(prefix):
        out = {}
        for name, r in honest_runs.items():
            base = name.rsplit("_s", 1)[0]
            seed = int(name.rsplit("_s", 1)[1])
            if base == prefix:
                out[seed] = r
        return out

    ref_seeds = per_seed("reference")
    stats = {"reference_seed_metrics": {}, "variants": {}, "sd": {f"{e}.{m}": v for (e, m), v in sds.items()}}
    for exp in EXPS:
        for m in METRICS:
            vals = [r["davigs"][exp][m] for r in ref_seeds.values()]
            stats["reference_seed_metrics"][f"{exp}.{m}"] = {
                "by_seed": {s: r["davigs"][exp][m] for s, r in ref_seeds.items()},
                "range": (max(vals) - min(vals)) if vals else 0.0,
            }
    variants = sorted({n.rsplit("_s", 1)[0] for n in honest_runs} - {"reference"})
    for v in variants:
        vs = per_seed(v)
        entry = {}
        for exp in EXPS:
            for m in METRICS:
                diffs = {s: vs[s]["davigs"][exp][m] - ref_seeds[s]["davigs"][exp][m] for s in vs if s in ref_seeds}
                view_diffs = []
                for s in diffs:
                    a = vs[s]["per_view"][exp] if vs[s]["per_view"] else []
                    b = ref_seeds[s]["per_view"][exp] if ref_seeds[s]["per_view"] else []
                    view_diffs += [x[m] - y[m] for x, y in zip(a, b)]
                mean_view = statistics.mean(view_diffs) if view_diffs else float("nan")
                se = statistics.stdev(view_diffs) / math.sqrt(len(view_diffs)) if len(view_diffs) > 1 else float("nan")
                entry[f"{exp}.{m}"] = {
                    "paired_diff_by_seed": diffs,
                    "per_view_paired_mean": mean_view,
                    "per_view_paired_t": mean_view / se if se == se and se > 0 else float("nan"),
                }
        entry["delta_rank_by_seed"] = {s: delta_rank(vs[s]["davigs"]) for s in vs}
        stats["variants"][v] = entry
    stats["reference_delta_rank_by_seed"] = {s: delta_rank(r["davigs"]) for s, r in ref_seeds.items()}

    sc_lines = ["| shortcut | davigs status | NV PSNR | AA PSNR | dR |", "|---|---|---:|---:|---:|"]
    for name, r in short.items():
        diag = r["diagnostics"].get("davigs", {})
        s = r["davigs"]
        status = diag.get("status", "-")
        if status == "failed":
            status += f" ({diag.get('reason', '')[:80]})"
        unscored = diag.get("unscored_summary")
        extra = ""
        if unscored:
            extra = f" [unscored: NV {unscored['glav_novel_view']['psnr']:.3f}, AA {unscored['glav_average_appearance']['psnr']:.3f}]"
        if s["glav_novel_view"] is None:
            sc_lines.append(f"| {name} | {status}{extra} | - | - | - |")
        else:
            sc_lines.append(
                f"| {name} | {status}{extra} | {s['glav_novel_view']['psnr']:.3f} | {s['glav_average_appearance']['psnr']:.3f} | {delta_rank(s):+.3f} |"
            )
    stats["shortcut_delta_rank"] = {n: (delta_rank(r["davigs"]) if r["davigs"]["glav_novel_view"] else None) for n, r in short.items()}
    (REPORT / "summary.json").write_text(json.dumps(stats, indent=1) + "\n")
    (REPORT / "summary.md").write_text("\n".join(lines) + "\n\n" + "\n".join(sc_lines) + "\n")
    print("\n".join(lines))
    print("\n".join(sc_lines))
    print(json.dumps(stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["honest", "shortcuts", "report", "all"])
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.stage in ("honest", "all"):
        honest()
    if args.stage in ("shortcuts", "all"):
        shortcuts()
    if args.stage in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
