#!/usr/bin/env python3
"""Visible train-slice (development scenes) evaluator for the GLAV-style appearance benchmark.

Scores every method directory under ``--outputs`` (default ``outputs/``) plus
every method the evaluator's roster requires, against evaluator-owned answers
in ``eval/train/answers.pt``. The only thing read from a method directory is
``predictions.pt``::

    {"novel_view":         {scene_id: [V, 3, H, W // 2] in [0, 1]},
     "average_appearance": {scene_id: [V, 3, H, W // 2] in [0, 1]}}

* ``glav_novel_view``: the paper's main protocol. Held-out captures may be
  fitted on their left half; PSNR / SSIM / LPIPS are measured on the right half.
* ``glav_average_appearance``: the paper's ablation protocol. The render made
  without any per-capture appearance state is colour-corrected (mip-NeRF 360
  procedure) to the average appearance of the camera over the training captures,
  then scored on the right half (the left half is a visible input). It is also the decoupling constraint: a method
  whose appearance-free render is worse than the evaluator's frozen vanilla-3DGS
  render by more than the roster margin fails the run.

Comparison methods (baselines, ablations) are scored from evaluator-owned frozen
predictions in ``eval/train/frozen/``; directories of the same name under
``outputs/`` are ignored. A missing, unreadable, malformed, non-finite or
out-of-range prediction, or a failed constraint, gives that method the worst
value of every metric in every experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SLICE = "train"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS = ("glav_novel_view", "glav_average_appearance")
METRICS = ("psnr", "ssim", "lpips")
WORST = {"psnr": 0.0, "ssim": -1.0, "lpips": 1.0}
RANGE_TOL = 1e-3
SECTIONS = (("novel_view", "right"), ("average_appearance", "avg"))


class InvalidPrediction(Exception):
    pass


# --------------------------------------------------------------------------- metrics
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.double() - b.double()) ** 2).mean())
    return min(100.0, -10.0 * math.log10(max(mse, 1e-10)))


_WIN = None


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    global _WIN
    if _WIN is None:
        x = torch.arange(11, dtype=torch.float64) - 5
        g = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        _WIN = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    a, b = a.double()[None], b.double()[None]
    mu_a, mu_b = F.conv2d(a, _WIN, groups=3), F.conv2d(b, _WIN, groups=3)
    s_aa = F.conv2d(a * a, _WIN, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, _WIN, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, _WIN, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())


def _load_weights(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # legacy-format official LPIPS file; checksum-verified by download.sh
        return torch.load(path, map_location="cpu", weights_only=False)


class LPIPSAlex(nn.Module):
    """LPIPS v0.1 with AlexNet features (Zhang et al. 2018), inputs in [0, 1]."""

    TAPS = (1, 4, 7, 9, 11)

    def __init__(self, weight_dir: Path):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        alex = _load_weights(weight_dir / "alexnet-owt-7be5be79.pth")
        feats = {k[len("features."):]: v for k, v in alex.items() if k.startswith("features.") and int(k.split(".")[1]) <= 10}
        self.net.load_state_dict(feats)
        lin = _load_weights(weight_dir / "alex_v0.1.pth")
        self.lins = [lin[f"lin{i}.model.1.weight"].float().cpu() for i in range(5)]
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = (2 * x.float() - 1 - self.shift) / self.scale
        fy = (2 * y.float() - 1 - self.shift) / self.scale
        total = torch.zeros(x.shape[0])
        tap = 0
        for i, layer in enumerate(self.net):
            fx, fy = layer(fx), layer(fy)
            if i in self.TAPS:
                nx = fx / (fx.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
                ny = fy / (fy.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
                total = total + ((nx - ny).pow(2) * self.lins[tap]).sum(1).mean((1, 2))
                tap += 1
        return total.clamp(0.0, 1.0)


def color_correct(img: torch.Tensor, ref: torch.Tensor, num_iters: int = 5, eps: float = 0.5 / 255) -> torch.Tensor:
    """mip-NeRF 360 colour correction: per-image quadratic colour warp of img to ref."""
    h, w = img.shape[-2:]
    cur = img.permute(1, 2, 0).reshape(-1, 3).double()
    refm = ref.permute(1, 2, 0).reshape(-1, 3).double()

    def unclipped(z):
        return (z >= eps) & (z <= 1 - eps)

    mask0 = unclipped(cur)
    for _ in range(num_iters):
        cols = [cur[:, c:c + 1] * cur[:, c:] for c in range(3)] + [cur, torch.ones_like(cur[:, :1])]
        a_mat = torch.cat(cols, dim=-1)
        warp = []
        for c in range(3):
            b = refm[:, c]
            mask = (mask0[:, c] & unclipped(cur[:, c]) & unclipped(b)).double()
            sol = torch.linalg.lstsq(a_mat * mask[:, None], (b * mask)[:, None], driver="gelsd").solution[:, 0]
            warp.append(sol)
        cur = (a_mat @ torch.stack(warp, dim=-1)).clamp(0.0, 1.0)
    if not torch.isfinite(cur).all():
        raise InvalidPrediction("colour correction produced non-finite values")
    return cur.reshape(h, w, 3).permute(2, 0, 1).float()


# --------------------------------------------------------------------------- loading
def load_prediction_file(path: Path, answers: dict) -> dict:
    if not path.is_file():
        raise InvalidPrediction("missing predictions.pt")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InvalidPrediction(f"unreadable predictions.pt ({type(exc).__name__})")
    if not isinstance(obj, dict):
        raise InvalidPrediction("predictions.pt must contain a dict")
    out = {}
    for section, key in SECTIONS:
        part = obj.get(section)
        if not isinstance(part, dict):
            raise InvalidPrediction(f"missing '{section}' section")
        out[section] = {}
        for sid, ans in answers["scenes"].items():
            t = part.get(sid)
            if not torch.is_tensor(t):
                raise InvalidPrediction(f"{section}: missing scene {sid}")
            if tuple(t.shape) != tuple(ans[key].shape):
                raise InvalidPrediction(f"{section}/{sid}: shape {tuple(t.shape)} != {tuple(ans[key].shape)}")
            if t.dtype == torch.uint8:
                t = t.float() / 255.0
            elif t.is_floating_point():
                t = t.float()
            else:
                raise InvalidPrediction(f"{section}/{sid}: dtype {t.dtype} not allowed")
            if not bool(torch.isfinite(t).all()):
                raise InvalidPrediction(f"{section}/{sid}: non-finite values")
            if float(t.min()) < -RANGE_TOL or float(t.max()) > 1 + RANGE_TOL:
                raise InvalidPrediction(f"{section}/{sid}: values outside [0, 1]")
            out[section][sid] = t.clamp(0.0, 1.0)
    return out


def score(pred: dict, answers: dict, lpips: LPIPSAlex):
    rows = {exp: [] for exp in EXPERIMENTS}
    for sid in sorted(answers["scenes"]):
        ans = answers["scenes"][sid]
        gt_r = ans["right"].float() / 255.0
        gt_a = ans["avg"].float()
        p_r = pred["novel_view"][sid]
        p_a = torch.stack([color_correct(p, g) for p, g in zip(pred["average_appearance"][sid], gt_a)])
        lp_r, lp_a = lpips(p_r, gt_r), lpips(p_a, gt_a)
        for i in range(gt_r.shape[0]):
            rows["glav_novel_view"].append(
                {"scene": sid, "view": i, "psnr": psnr(p_r[i], gt_r[i]), "ssim": ssim(p_r[i], gt_r[i]), "lpips": float(lp_r[i])}
            )
            rows["glav_average_appearance"].append(
                {"scene": sid, "view": i, "psnr": psnr(p_a[i], gt_a[i]), "ssim": ssim(p_a[i], gt_a[i]), "lpips": float(lp_a[i])}
            )
    summary = {exp: {m: sum(r[m] for r in rs) / len(rs) for m in METRICS} for exp, rs in rows.items()}
    return summary, rows


# --------------------------------------------------------------------------- driver
def evaluate(outputs: Path, answers_path: Path, frozen_dir: Path, roster_path: Path, weight_dir: Path):
    answers = torch.load(answers_path, map_location="cpu", weights_only=True)
    roster = json.loads(roster_path.read_text())
    lpips = LPIPSAlex(weight_dir)
    results = {exp: {} for exp in EXPERIMENTS}
    diagnostics, details, fixed = {}, {}, {}

    for name, exps in roster["comparisons"].items():
        pred = load_prediction_file(frozen_dir / f"{name}.pt", answers)  # evaluator asset: errors are fatal
        summary, rows = score(pred, answers, lpips)
        fixed[name] = summary
        for exp in exps:
            results[exp][name] = summary[exp]
        diagnostics[name] = {"status": "comparison: evaluator-owned frozen predictions"}
        details[name] = rows

    gate = roster["gate"]
    threshold = fixed[gate["reference_method"]][gate["experiment"]][gate["metric"]] - float(gate["margin"])
    on_disk = []
    if outputs.is_dir():
        on_disk = sorted(p.name for p in outputs.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
    for name in sorted(set(on_disk) | set(roster["required"])):
        if name in roster["comparisons"]:
            diagnostics[name]["ignored_outputs_dir"] = True
            continue
        try:
            pred = load_prediction_file(outputs / name / "predictions.pt", answers)
            summary, rows = score(pred, answers, lpips)
            value = summary[gate["experiment"]][gate["metric"]]
            if not value >= threshold:
                diagnostics[name] = {"unscored_summary": summary}
                raise InvalidPrediction(
                    f"constraint failed: {gate['experiment']}.{gate['metric']} = {value:.3f} < {threshold:.3f} "
                    f"({gate['reference_method']} - {gate['margin']})"
                )
            for exp in EXPERIMENTS:
                results[exp][name] = summary[exp]
            diagnostics[name] = {"status": "ok", "constraint_value": value, "constraint_threshold": threshold}
            details[name] = rows
        except Exception as exc:
            if not isinstance(exc, InvalidPrediction):
                traceback.print_exc()
            for exp in EXPERIMENTS:
                results[exp][name] = dict(WORST)
            diagnostics.setdefault(name, {})
            diagnostics[name].update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})

    payload = {
        "slice": SLICE,
        "experiments": {exp: {"results": results[exp]} for exp in EXPERIMENTS},
        "_diagnostics": diagnostics,
    }
    return payload, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    ap.add_argument("--scores", type=Path, default=ROOT / "scoring" / "scores_train.json")
    ap.add_argument("--details", type=Path, default=None, help="optional per-view metric dump")
    args = ap.parse_args()
    payload, details = evaluate(
        args.outputs, HERE / "answers.pt", HERE / "frozen", HERE / "roster.json", ROOT / "data" / "downloads" / "lpips"
    )
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(json.dumps(details) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "_diagnostics"}, indent=2, sort_keys=True))
    for name, diag in payload["_diagnostics"].items():
        print(f"[{SLICE}] {name}: {diag.get('status')} {diag.get('reason', '')}", file=sys.stderr)


if __name__ == "__main__":
    main()
