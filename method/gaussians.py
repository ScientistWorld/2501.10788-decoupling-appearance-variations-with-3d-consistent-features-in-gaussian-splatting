"""3D Gaussian Splatting (Kerbl et al. 2023) in plain PyTorch.

Anisotropic Gaussians (scale + rotation quaternion), spherical-harmonic colours
(degree 3, raised progressively), EWA projection with the 0.3-pixel low-pass,
depth-sorted front-to-back alpha compositing, rendered mean depth (normalised
by accumulated weight, as used by DAVIGS for back-projection), and adaptive
density control (clone / split / prune, periodic opacity reset) with
per-parameter Adam learning rates from the 3DGS reference implementation.

The rasteriser is sparse PyTorch instead of tile-based CUDA: every Gaussian is
evaluated on the pixels of its 3-sigma footprint (as 3DGS does via tiles),
(Gaussian, pixel) pairs are sorted per pixel by depth, and transmittance is a
segmented exclusive cumulative product, so compositing is exact front-to-back
alpha blending.
"""

from __future__ import annotations

import math

import torch
from torch import nn

PATCH_BUCKETS = (3, 6, 12, 24, 48, 96)  # half-widths (px) of per-Gaussian evaluation patches
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: [N, 3, 16], dirs: [N, 3] unit vectors -> [N, 3]."""
    result = SH_C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - SH_C1 * y * sh[..., 1] + SH_C1 * z * sh[..., 2] - SH_C1 * x * sh[..., 3]
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + SH_C2[0] * xy * sh[..., 4]
                + SH_C2[1] * yz * sh[..., 5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + SH_C2[3] * xz * sh[..., 7]
                + SH_C2[4] * (xx - yy) * sh[..., 8]
            )
            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + SH_C3[1] * xy * z * sh[..., 10]
                    + SH_C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + SH_C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + SH_C3[5] * z * (xx - yy) * sh[..., 14]
                    + SH_C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )
    return result


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class Camera:
    def __init__(self, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, height: int, width: int, device):
        self.R = R.float().to(device)
        self.t = t.float().to(device)
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
        self.H, self.W = int(height), int(width)
        self.center = -(self.R.T @ self.t)
        self.px = torch.arange(self.W, device=device, dtype=torch.float32).repeat(self.H)
        self.py = torch.arange(self.H, device=device, dtype=torch.float32).repeat_interleave(self.W)


def knn_mean_dist2(pts: torch.Tensor, k: int = 3) -> torch.Tensor:
    out = []
    for start in range(0, pts.shape[0], 2048):
        d2 = torch.cdist(pts[start:start + 2048], pts).pow(2)
        idx = torch.arange(start, min(start + 2048, pts.shape[0]), device=pts.device)
        d2[torch.arange(idx.numel(), device=pts.device), idx] = float("inf")
        out.append(d2.topk(k, dim=1, largest=False).values.mean(dim=1))
    return torch.cat(out)


class GaussianModel:
    NAMES = ("means", "sh_dc", "sh_rest", "opacity", "log_scales", "quats")

    def __init__(self, points: torch.Tensor, colors: torch.Tensor, extent: float, device, max_gaussians: int = 6000):
        pts = points.float().to(device)
        n = pts.shape[0]
        d2 = knn_mean_dist2(pts).clamp_min(1e-7)
        tensors = {
            "means": pts,
            "sh_dc": (colors.float().to(device) - 0.5) / SH_C0,
            "sh_rest": torch.zeros(n, 15, 3, device=device),
            "opacity": torch.logit(torch.full((n,), 0.1, device=device)),
            "log_scales": torch.log(torch.sqrt(d2))[:, None].repeat(1, 3),
            "quats": torch.cat([torch.ones(n, 1, device=device), torch.zeros(n, 3, device=device)], dim=1),
        }
        self.p = {k: nn.Parameter(v.contiguous()) for k, v in tensors.items()}
        self.extent = float(extent)
        self.device = device
        self.max_gaussians = int(max_gaussians)
        self.percent_dense = 0.01
        self.pos_lr_init = 1.6e-4 * self.extent
        self.pos_lr_final = 1.6e-6 * self.extent
        lrs = {
            "means": self.pos_lr_init,
            "sh_dc": 0.0025,
            "sh_rest": 0.0025 / 20.0,
            "opacity": 0.025,
            "log_scales": 0.005,
            "quats": 0.001,
        }
        self.opt = torch.optim.Adam(
            [{"params": [self.p[k]], "lr": lrs[k], "name": k} for k in self.NAMES], lr=0.0, eps=1e-15
        )
        self.reset_stats()

    # ------------------------------------------------------------------ accessors
    @property
    def n(self) -> int:
        return self.p["means"].shape[0]

    def sh(self) -> torch.Tensor:
        return torch.cat([self.p["sh_dc"][:, None, :], self.p["sh_rest"]], dim=1)  # [N, 16, 3]

    def set_position_lr(self, frac: float) -> None:
        frac = min(max(frac, 0.0), 1.0)
        lr = math.exp(math.log(self.pos_lr_init) * (1 - frac) + math.log(self.pos_lr_final) * frac)
        for g in self.opt.param_groups:
            if g["name"] == "means":
                g["lr"] = lr

    def reset_stats(self) -> None:
        self.grad_accum = torch.zeros(self.n, device=self.device)
        self.denom = torch.zeros(self.n, device=self.device)

    @torch.no_grad()
    def export(self) -> dict:
        """The scene as plain tensors (the format the evaluator renders itself)."""
        return {
            "means": self.p["means"].detach().float().cpu().contiguous(),
            "scales": torch.exp(self.p["log_scales"].detach()).float().cpu().contiguous(),
            "quats": self.p["quats"].detach().float().cpu().contiguous(),
            "opacity": torch.sigmoid(self.p["opacity"].detach()).float().cpu().contiguous(),
            "sh": self.sh().detach().float().cpu().contiguous(),
        }

    # ------------------------------------------------------------------ rendering
    def render(self, cam: Camera, sh_degree: int, bg: torch.Tensor, track: bool = False):
        means = self.p["means"]
        pc = means @ cam.R.T + cam.t
        idx = torch.nonzero(pc[:, 2] > 0.2).squeeze(1)
        empty = (
            bg[:, None, None].expand(3, cam.H, cam.W).clone(),
            torch.zeros(cam.H, cam.W, device=means.device),
            torch.zeros(cam.H, cam.W, device=means.device),
            None,
        )
        if idx.numel() == 0:
            return empty
        p = pc[idx]
        z = p[:, 2]
        u = cam.fx * p[:, 0] / z + cam.cx
        v = cam.fy * p[:, 1] / z + cam.cy
        mean2d = torch.stack([u, v], dim=-1)
        if track and mean2d.requires_grad:
            mean2d.retain_grad()
        u, v = mean2d[:, 0], mean2d[:, 1]  # route all screen-space use through mean2d
        rot = quat_to_rotmat(self.p["quats"][idx])
        s = torch.exp(self.p["log_scales"][idx])
        m = rot * s[:, None, :]
        cov_c = cam.R @ (m @ m.transpose(1, 2)) @ cam.R.T
        limx = 1.3 * 0.5 * cam.W / cam.fx
        limy = 1.3 * 0.5 * cam.H / cam.fy
        tx = (p[:, 0] / z).clamp(-limx, limx) * z
        ty = (p[:, 1] / z).clamp(-limy, limy) * z
        zero = torch.zeros_like(z)
        jac = torch.stack(
            [
                torch.stack([cam.fx / z, zero, -cam.fx * tx / (z * z)], dim=-1),
                torch.stack([zero, cam.fy / z, -cam.fy * ty / (z * z)], dim=-1),
            ],
            dim=1,
        )
        cov2 = jac @ cov_c @ jac.transpose(1, 2)
        a = cov2[:, 0, 0] + 0.3
        b = cov2[:, 0, 1]
        c = cov2[:, 1, 1] + 0.3
        det = (a * c - b * b).clamp_min(1e-10)
        mid = 0.5 * (a + c)
        radius = 3.0 * torch.sqrt(mid + torch.sqrt((mid * mid - det).clamp_min(0.1)))
        on = (u + radius > -0.5) & (u - radius < cam.W - 0.5) & (v + radius > -0.5) & (v - radius < cam.H - 0.5)
        keep = torch.nonzero(on).squeeze(1)
        if keep.numel() == 0:
            return empty
        vis = idx[keep]
        dirs = means[vis] - cam.center[None, :]
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rgb = (eval_sh(sh_degree, self.sh()[vis].transpose(1, 2), dirs) + 0.5).clamp_min(0.0)
        opac = torch.sigmoid(self.p["opacity"][vis])
        zk = z[keep]
        order = torch.argsort(zk)
        uk, vk = u[keep][order], v[keep][order]
        ia = (c / det)[keep][order]
        ib = (-b / det)[keep][order]
        ic = (a / det)[keep][order]
        zk, rgb, opac = zk[order], rgb[order], opac[order]

        rk = radius[keep][order]

        # Sparse rasterisation: each Gaussian is evaluated only on the pixels of
        # its 3-sigma footprint (patch sizes bucketed), then (Gaussian, pixel)
        # pairs are sorted by pixel and depth and alpha-composited front to back
        # with a segmented exclusive cumulative sum of log(1 - alpha).
        need = torch.ceil(rk.detach() + 0.5)
        cu = torch.round(uk.detach()).long()
        cv = torch.round(vk.detach()).long()
        assigned = torch.zeros_like(need, dtype=torch.bool)
        g_list, p_list, a_list = [], [], []
        for bi, rad in enumerate(PATCH_BUCKETS):
            sel = ~assigned if bi == len(PATCH_BUCKETS) - 1 else (need <= rad) & ~assigned
            assigned |= sel
            gb = torch.nonzero(sel).squeeze(1)
            if gb.numel() == 0:
                continue
            off = torch.arange(-rad, rad + 1, device=means.device)
            oy, ox = torch.meshgrid(off, off, indexing="ij")
            px = cu[gb][:, None] + ox.reshape(1, -1)
            py = cv[gb][:, None] + oy.reshape(1, -1)
            inb = (px >= 0) & (px < cam.W) & (py >= 0) & (py < cam.H)
            gi = gb[:, None].expand_as(px)[inb]
            pxv, pyv = px[inb], py[inb]
            dx = pxv.to(uk.dtype) - uk[gi]
            dy = pyv.to(vk.dtype) - vk[gi]
            power = -0.5 * (ia[gi] * dx * dx + ic[gi] * dy * dy) - ib[gi] * dx * dy
            alpha = (opac[gi] * torch.exp(power.clamp_max(0.0))).clamp(max=0.99)
            ok = alpha.detach() >= 1.0 / 255.0
            g_list.append(gi[ok])
            p_list.append(pyv[ok] * cam.W + pxv[ok])
            a_list.append(alpha[ok])
        if not g_list:
            return empty
        g = torch.cat(g_list)
        pix = torch.cat(p_list)
        al = torch.cat(a_list)
        if g.numel() == 0:
            return empty
        srt = torch.argsort(pix * uk.shape[0] + g)
        g, pix, al = g[srt], pix[srt], al[srt]
        log1m = torch.log1p(-al).double()
        before = torch.cumsum(log1m, dim=0) - log1m
        m = g.shape[0]
        is_start = torch.ones(m, dtype=torch.bool, device=means.device)
        is_start[1:] = pix[1:] != pix[:-1]
        start = torch.where(is_start, torch.arange(m, device=means.device), torch.zeros_like(pix)).cummax(dim=0).values
        trans = torch.exp(before - before[start]).to(al.dtype)
        w = al * trans
        n_pix = cam.H * cam.W
        color = torch.zeros(n_pix, 3, device=means.device, dtype=w.dtype).index_add(0, pix, w[:, None] * rgb[g])
        acc = torch.zeros(n_pix, device=means.device, dtype=w.dtype).index_add(0, pix, w)
        dnum = torch.zeros(n_pix, device=means.device, dtype=w.dtype).index_add(0, pix, w * zk[g])
        log_t = torch.zeros(n_pix, device=means.device, dtype=log1m.dtype).index_add(0, pix, log1m)
        color = color + torch.exp(log_t).to(w.dtype)[:, None] * bg[None, :]
        depth = dnum / acc.clamp_min(1e-6)
        image = color.T.reshape(3, cam.H, cam.W)
        info = {"mean2d": mean2d, "keep": keep, "vis": vis, "W": cam.W, "H": cam.H}
        return image, depth.reshape(cam.H, cam.W), acc.reshape(cam.H, cam.W), info

    # ------------------------------------------------------------------ density control
    def add_densify_stats(self, info) -> None:
        if info is None or info["mean2d"].grad is None:
            return
        g = info["mean2d"].grad[info["keep"]]
        gn = torch.sqrt((g[:, 0] * info["W"] * 0.5) ** 2 + (g[:, 1] * info["H"] * 0.5) ** 2)
        self.grad_accum.index_add_(0, info["vis"], gn)
        self.denom.index_add_(0, info["vis"], torch.ones_like(gn))

    def _cat(self, ext: dict) -> None:
        for group in self.opt.param_groups:
            name = group["name"]
            old = group["params"][0]
            add = ext[name].to(old.dtype)
            new = nn.Parameter(torch.cat([old.detach(), add], dim=0).contiguous())
            st = self.opt.state.pop(old, None)
            if st:
                st["exp_avg"] = torch.cat([st["exp_avg"], torch.zeros_like(add)], dim=0)
                st["exp_avg_sq"] = torch.cat([st["exp_avg_sq"], torch.zeros_like(add)], dim=0)
                self.opt.state[new] = st
            group["params"][0] = new
            self.p[name] = new

    def _prune(self, keep: torch.Tensor) -> None:
        for group in self.opt.param_groups:
            name = group["name"]
            old = group["params"][0]
            new = nn.Parameter(old.detach()[keep].contiguous())
            st = self.opt.state.pop(old, None)
            if st:
                st["exp_avg"] = st["exp_avg"][keep]
                st["exp_avg_sq"] = st["exp_avg_sq"][keep]
                self.opt.state[new] = st
            group["params"][0] = new
            self.p[name] = new

    def _replace(self, name: str, value: torch.Tensor) -> None:
        for group in self.opt.param_groups:
            if group["name"] != name:
                continue
            old = group["params"][0]
            new = nn.Parameter(value.contiguous())
            st = self.opt.state.pop(old, None)
            if st:
                st["exp_avg"] = torch.zeros_like(value)
                st["exp_avg_sq"] = torch.zeros_like(value)
                self.opt.state[new] = st
            group["params"][0] = new
            self.p[name] = new

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold: float, min_opacity: float, prune_big: bool) -> None:
        n0 = self.n
        grads = self.grad_accum / self.denom.clamp_min(1.0)
        scales = torch.exp(self.p["log_scales"]).max(dim=1).values
        sel = grads >= grad_threshold
        small = scales <= self.percent_dense * self.extent
        clone, split = sel & small, sel & ~small
        budget = self.max_gaussians - n0
        if int(clone.sum()) + int(split.sum()) > budget:
            allowed = torch.zeros_like(sel)
            if budget > 0:
                cand = torch.nonzero(sel).squeeze(1)
                allowed[cand[torch.argsort(grads[cand], descending=True)[:budget]]] = True
            clone &= allowed
            split &= allowed
        parts = {k: [self.p[k].detach()[clone]] for k in self.NAMES}
        if bool(split.any()):
            s = torch.exp(self.p["log_scales"].detach()[split]).repeat(2, 1)
            rots = quat_to_rotmat(self.p["quats"].detach()[split]).repeat(2, 1, 1)
            offsets = (rots @ (torch.randn_like(s) * s)[..., None]).squeeze(-1)
            parts["means"].append(self.p["means"].detach()[split].repeat(2, 1) + offsets)
            parts["log_scales"].append(torch.log(s / 1.6))
            for k in ("sh_dc", "sh_rest", "opacity", "quats"):
                t = self.p[k].detach()[split]
                parts[k].append(t.repeat(2, *([1] * (t.dim() - 1))))
        ext = {k: torch.cat(v, dim=0) for k, v in parts.items()}
        if ext["means"].shape[0] > 0:
            self._cat(ext)
        prune = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        prune[:n0] = split
        prune |= torch.sigmoid(self.p["opacity"].detach()) < min_opacity
        if prune_big:
            prune |= torch.exp(self.p["log_scales"].detach()).max(dim=1).values > 0.1 * self.extent
        if bool(prune.any()) and int((~prune).sum()) > 0:
            self._prune(~prune)
        self.reset_stats()

    @torch.no_grad()
    def reset_opacity(self) -> None:
        cur = torch.sigmoid(self.p["opacity"].detach())
        self._replace("opacity", torch.logit(torch.minimum(cur, torch.full_like(cur, 0.01))))
