#!/usr/bin/env python3
"""Visible train-slice (development scenes) evaluator for the GLAV-style appearance benchmark.

Scores every method directory under ``--outputs`` (default ``outputs/``) plus
every method the evaluator's roster requires, against evaluator-owned answers
in ``eval/train/answers.pt``. The only thing read from a method directory is
``predictions.pt``::

    {"novel_view": {scene_id: [6, 3, H, W // 2] in [0, 1]},   # held-out captures, right halves
     "scene":      {scene_id: {"means": [N, 3], "scales": [N, 3] (> 0), "quats": [N, 4] (w, x, y, z),
                               "opacity": [N] in [0, 1], "sh": [N, 16, 3] (degree-3 SH colour)}}}

* ``glav_novel_view``: the paper's main protocol. Held-out captures may be
  fitted on their left half; PSNR / SSIM / LPIPS are measured on the right half.
* ``glav_average_appearance``: the paper's colour-corrected (average-appearance)
  protocol. The evaluator itself renders the submitted 3D Gaussian scene (its
  own 3DGS rasteriser, black background, no appearance model, so nothing
  per-capture or coupled can enter) at six canonical-only cameras (no capture
  exists there); each render is colour-corrected by the
  evaluator (mip-NeRF 360 quadratic warp) to the camera's average appearance over
  the training capture conditions and scored on the full frame. It is also the
  decoupling constraint: a method whose appearance-free PSNR falls more than the
  roster margin below the evaluator's frozen reference reconstruction fails the run.

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
import os
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
MAX_GAUSSIANS = 60000          # keeps the hidden test light (10x the reference budget)
MAX_PAIRS_PER_IMAGE = 120_000_000  # Gaussian-pixel evaluations per render (guard against pathological scenes)
SCENE_KEYS = {"means": (3,), "scales": (3,), "quats": (4,), "opacity": (), "sh": (16, 3)}


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


# --------------------------------------------------------------------------- scene rendering
# Forward-only 3D Gaussian Splatting rasteriser (Kerbl et al. 2023): EWA projection
# with the 0.3 px low-pass, degree-3 SH colour, depth-sorted front-to-back alpha
# compositing over a black background. Evaluator-owned copy; nothing is imported
# from method/.
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
PATCH_BUCKETS = (3, 6, 12, 24, 48, 96)


def _eval_sh(sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: [N, 3, 16], dirs: [N, 3] -> [N, 3]."""
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    r = SH_C0 * sh[..., 0] - SH_C1 * y * sh[..., 1] + SH_C1 * z * sh[..., 2] - SH_C1 * x * sh[..., 3]
    r = (r + SH_C2[0] * xy * sh[..., 4] + SH_C2[1] * yz * sh[..., 5] + SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
         + SH_C2[3] * xz * sh[..., 7] + SH_C2[4] * (xx - yy) * sh[..., 8])
    r = (r + SH_C3[0] * y * (3 * xx - yy) * sh[..., 9] + SH_C3[1] * xy * z * sh[..., 10]
         + SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
         + SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] + SH_C3[5] * z * (xx - yy) * sh[..., 14]
         + SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15])
    return r


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
         2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
         2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


@torch.no_grad()
def render_scene(g: dict, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    R, t = R.float(), t.float()
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    image = torch.zeros(3, height, width)
    means = g["means"]
    pc = means @ R.T + t
    idx = torch.nonzero(pc[:, 2] > 0.2).squeeze(1)
    if idx.numel() == 0:
        return image
    p = pc[idx]
    z = p[:, 2]
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    rot = _quat_to_rotmat(g["quats"][idx])
    m = rot * g["scales"][idx][:, None, :]
    cov_c = R @ (m @ m.transpose(1, 2)) @ R.T
    limx, limy = 1.3 * 0.5 * width / fx, 1.3 * 0.5 * height / fy
    tx = (p[:, 0] / z).clamp(-limx, limx) * z
    ty = (p[:, 1] / z).clamp(-limy, limy) * z
    zero = torch.zeros_like(z)
    jac = torch.stack([torch.stack([fx / z, zero, -fx * tx / (z * z)], -1), torch.stack([zero, fy / z, -fy * ty / (z * z)], -1)], 1)
    cov2 = jac @ cov_c @ jac.transpose(1, 2)
    a, b, c = cov2[:, 0, 0] + 0.3, cov2[:, 0, 1], cov2[:, 1, 1] + 0.3
    det = (a * c - b * b).clamp_min(1e-10)
    mid = 0.5 * (a + c)
    radius = 3.0 * torch.sqrt(mid + torch.sqrt((mid * mid - det).clamp_min(0.1)))
    on = (u + radius > -0.5) & (u - radius < width - 0.5) & (v + radius > -0.5) & (v - radius < height - 0.5)
    keep = torch.nonzero(on).squeeze(1)
    if keep.numel() == 0:
        return image
    vis = idx[keep]
    center = -(R.T @ t)
    dirs = means[vis] - center[None, :]
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    rgb = (_eval_sh(g["sh"][vis].transpose(1, 2), dirs) + 0.5).clamp_min(0.0)
    opac = g["opacity"][vis]
    zk = z[keep]
    order = torch.argsort(zk)
    uk, vk = u[keep][order], v[keep][order]
    ia, ib, ic = (c / det)[keep][order], (-b / det)[keep][order], (a / det)[keep][order]
    zk, rgb, opac, rk = zk[order], rgb[order], opac[order], radius[keep][order]
    need = torch.ceil(rk + 0.5)
    cu, cv = torch.round(uk).long(), torch.round(vk).long()
    assigned = torch.zeros_like(need, dtype=torch.bool)
    budget = 0
    g_list, p_list, a_list = [], [], []
    for bi, rad in enumerate(PATCH_BUCKETS):
        sel = ~assigned if bi == len(PATCH_BUCKETS) - 1 else (need <= rad) & ~assigned
        assigned |= sel
        gb_all = torch.nonzero(sel).squeeze(1)
        area = (2 * rad + 1) ** 2
        budget += gb_all.numel() * area
        if budget > MAX_PAIRS_PER_IMAGE:
            raise InvalidPrediction("scene too expensive to render (Gaussian footprints too large or too many)")
        off = torch.arange(-rad, rad + 1)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        ox, oy = ox.reshape(1, -1), oy.reshape(1, -1)
        chunk = max(1, 4_000_000 // area)
        for s0 in range(0, gb_all.numel(), chunk):
            gb = gb_all[s0:s0 + chunk]
            px, py = cu[gb][:, None] + ox, cv[gb][:, None] + oy
            inb = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            gi = gb[:, None].expand_as(px)[inb]
            pxv, pyv = px[inb], py[inb]
            dx, dy = pxv.float() - uk[gi], pyv.float() - vk[gi]
            power = -0.5 * (ia[gi] * dx * dx + ic[gi] * dy * dy) - ib[gi] * dx * dy
            alpha = (opac[gi] * torch.exp(power.clamp_max(0.0))).clamp(max=0.99)
            ok = alpha >= 1.0 / 255.0
            g_list.append(gi[ok])
            p_list.append(pyv[ok] * width + pxv[ok])
            a_list.append(alpha[ok])
    gg, pix, al = torch.cat(g_list), torch.cat(p_list), torch.cat(a_list)
    if gg.numel() == 0:
        return image
    srt = torch.argsort(pix * uk.shape[0] + gg)
    gg, pix, al = gg[srt], pix[srt], al[srt]
    log1m = torch.log1p(-al).double()
    before = torch.cumsum(log1m, dim=0) - log1m
    n = gg.shape[0]
    is_start = torch.ones(n, dtype=torch.bool)
    is_start[1:] = pix[1:] != pix[:-1]
    start = torch.where(is_start, torch.arange(n), torch.zeros_like(pix)).cummax(dim=0).values
    w = al * torch.exp(before - before[start]).float()
    color = torch.zeros(height * width, 3).index_add(0, pix, w[:, None] * rgb[gg])
    return color.T.reshape(3, height, width)


def check_scene(sid: str, g) -> dict:
    if not isinstance(g, dict):
        raise InvalidPrediction(f"scene/{sid}: must be a dict of tensors")
    out, n = {}, None
    for key, tail in SCENE_KEYS.items():
        x = g.get(key)
        if not torch.is_tensor(x) or not x.is_floating_point():
            raise InvalidPrediction(f"scene/{sid}: '{key}' must be a float tensor")
        if n is None:
            n = x.shape[0] if x.dim() > 0 else 0
        if tuple(x.shape) != (n, *tail):
            raise InvalidPrediction(f"scene/{sid}: '{key}' shape {tuple(x.shape)} != {(n, *tail)}")
        x = x.float()
        if not bool(torch.isfinite(x).all()):
            raise InvalidPrediction(f"scene/{sid}: '{key}' has non-finite values")
        out[key] = x
    if not 1 <= n <= MAX_GAUSSIANS:
        raise InvalidPrediction(f"scene/{sid}: {n} Gaussians (allowed 1..{MAX_GAUSSIANS})")
    if float(out["opacity"].min()) < -RANGE_TOL or float(out["opacity"].max()) > 1 + RANGE_TOL:
        raise InvalidPrediction(f"scene/{sid}: opacity outside [0, 1]")
    out["opacity"] = out["opacity"].clamp(0.0, 1.0)
    if float(out["scales"].min()) <= 0 or float(out["scales"].max()) > 1e3:
        raise InvalidPrediction(f"scene/{sid}: scales must lie in (0, 1e3]")
    if float(out["quats"].norm(dim=1).min()) < 1e-8:
        raise InvalidPrediction(f"scene/{sid}: zero-norm quaternion")
    return out


def render_canonical(g: dict, ans: dict) -> torch.Tensor:
    h, w = ans["canon"].shape[-2:]
    imgs = [render_scene(g, ans["canon_R"][i], ans["canon_t"][i], ans["canon_K"][i], h, w) for i in range(ans["canon_R"].shape[0])]
    return torch.stack(imgs).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- loading
def _check_images(name: str, t, shape) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise InvalidPrediction(f"{name}: missing")
    if tuple(t.shape) != tuple(shape):
        raise InvalidPrediction(f"{name}: shape {tuple(t.shape)} != {tuple(shape)}")
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    elif t.is_floating_point():
        t = t.float()
    else:
        raise InvalidPrediction(f"{name}: dtype {t.dtype} not allowed")
    if not bool(torch.isfinite(t).all()):
        raise InvalidPrediction(f"{name}: non-finite values")
    if float(t.min()) < -RANGE_TOL or float(t.max()) > 1 + RANGE_TOL:
        raise InvalidPrediction(f"{name}: values outside [0, 1]")
    return t.clamp(0.0, 1.0)


def _read(path: Path):
    if not path.is_file():
        raise InvalidPrediction("missing predictions.pt")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InvalidPrediction(f"unreadable predictions.pt ({type(exc).__name__})")
    if not isinstance(obj, dict):
        raise InvalidPrediction("predictions.pt must contain a dict")
    return obj


def load_prediction_file(path: Path, answers: dict) -> dict:
    """A method's predictions: novel-view images plus its 3D Gaussian scene, which
    the evaluator renders at the canonical cameras itself."""
    obj = _read(path)
    nv, sc = obj.get("novel_view"), obj.get("scene")
    if not isinstance(nv, dict):
        raise InvalidPrediction("missing 'novel_view' section")
    if not isinstance(sc, dict):
        raise InvalidPrediction("missing 'scene' section")
    out = {"novel_view": {}, "average_appearance": {}}
    for sid, ans in answers["scenes"].items():
        out["novel_view"][sid] = _check_images(f"novel_view/{sid}", nv.get(sid), ans["right"].shape)
        if sid not in sc:
            raise InvalidPrediction(f"scene: missing scene {sid}")
        out["average_appearance"][sid] = render_canonical(check_scene(sid, sc[sid]), ans)
    return out


def load_frozen(path: Path, answers: dict) -> dict:
    """Evaluator-owned frozen comparison: images (canonical renders were made by
    this evaluator's renderer when the comparison was frozen)."""
    obj = _read(path)
    out = {"novel_view": {}, "average_appearance": {}}
    for sid, ans in answers["scenes"].items():
        out["novel_view"][sid] = _check_images(f"frozen novel_view/{sid}", obj["novel_view"].get(sid), ans["right"].shape)
        out["average_appearance"][sid] = _check_images(f"frozen average_appearance/{sid}", obj["average_appearance"].get(sid), ans["canon"].shape)
    return out


def score(pred: dict, answers: dict, lpips: LPIPSAlex):
    rows = {exp: [] for exp in EXPERIMENTS}
    for sid in sorted(answers["scenes"]):
        ans = answers["scenes"][sid]
        gt_r = ans["right"].float() / 255.0
        gt_c = ans["canon"].float()
        p_r = pred["novel_view"][sid]
        p_c = torch.stack([color_correct(p, g) for p, g in zip(pred["average_appearance"][sid], gt_c)])
        lp_r, lp_c = lpips(p_r, gt_r), lpips(p_c, gt_c)
        for i in range(gt_r.shape[0]):
            rows["glav_novel_view"].append(
                {"scene": sid, "view": i, "psnr": psnr(p_r[i], gt_r[i]), "ssim": ssim(p_r[i], gt_r[i]), "lpips": float(lp_r[i])}
            )
        for i in range(gt_c.shape[0]):
            rows["glav_average_appearance"].append(
                {"scene": sid, "view": i, "psnr": psnr(p_c[i], gt_c[i]), "ssim": ssim(p_c[i], gt_c[i]), "lpips": float(lp_c[i])}
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
        pred = load_frozen(frozen_dir / f"{name}.pt", answers)  # evaluator asset: errors are fatal
        summary, rows = score(pred, answers, lpips)
        fixed[name] = summary
        for exp in exps:
            results[exp][name] = summary[exp]
        diagnostics[name] = {"status": "comparison: evaluator-owned frozen predictions"}
        details[name] = rows

    # Decoupling constraint: evaluator-owned frozen reference reconstruction (never read from outputs/).
    gate = roster["gate"]
    gate_pred = load_frozen(frozen_dir / f"{gate['reference_file']}.pt", answers)
    gate_summary, _ = score(gate_pred, answers, lpips)
    gate_value = gate_summary[gate["experiment"]][gate["metric"]]
    threshold = gate_value - float(gate["margin"])
    diagnostics["_gate"] = {
        "status": "gate reference (evaluator-owned frozen predictions)",
        "reference_file": gate["reference_file"],
        "reference_value": gate_value,
        "margin": float(gate["margin"]),
        "threshold": threshold,
    }
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
                    f"(frozen {gate['reference_file']} - {gate['margin']})"
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
    torch.set_num_threads(max(1, int(os.environ.get("GLAV_EVAL_THREADS", "2"))))
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
