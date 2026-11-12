#!/usr/bin/env python3
"""Owner-side generator for the GLAV-style appearance-variation benchmark.

This file lives in ``eval/test/`` because, together with the seeds below, it can
regenerate the hidden test answers. Nothing outside ``eval/test/`` imports it.

It writes
  * ``data/glav_syn/manifest.json`` and ``data/glav_syn/<scene>.pt``: the inputs a
    method may see (training images, cameras, left halves of held-out captures,
    and a sparse SfM-like point cloud),
  * ``eval/train/answers.pt``: ground truth for the visible development scenes,
  * ``eval/test/answers.pt``: ground truth for the hidden test scenes.

Scenes are ray-traced from analytic, procedurally textured geometry (boxes and
spheres inside a walled courtyard), not from Gaussians, and supersampled 2x2.
Every capture has its own appearance: a capture session (day / golden hour /
overcast / night) fixes a moving sun with cast shadows, sky/ambient light and
which point lamps are lit (local, 3D-consistent variation); each capture then
adds per-image jitter and a camera ISP (auto-exposure with compensation, white
balance, a filmic tone curve, gamma, saturation and black level).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "glav_syn"
TRAIN_ANSWERS = ROOT / "eval" / "train" / "answers.pt"
TEST_ANSWERS = ROOT / "eval" / "test" / "answers.pt"

H, W = 96, 128
SS = 2
FOCAL = (W / 2.0) / math.tan(math.radians(30.0))  # 60 degree horizontal field of view
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
N_VIEWS = 36
HELDOUT_IDX = [2, 8, 14, 20, 26, 32]
N_INIT = 3000
EPS = 1e-6
DT = torch.float64

DEV_SEEDS = {"dev_00": 1101, "dev_01": 2202, "dev_02": 3303}
TEST_SEEDS = {
    "test_00": 2515672036013325220,
    "test_01": 928762439827652497,
    "test_02": 4581806996186167911,
    "test_03": 538388288356111426,
    "test_04": 1165364637827746705,
    "test_05": 628636198225082335,
}


# --------------------------------------------------------------------------- random helpers
def uf(g, lo, hi):
    return lo + (hi - lo) * torch.rand(1, generator=g, dtype=DT).item()


def uv(g, lo, hi, *shape):
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=DT)


def nv(g, *shape):
    return torch.randn(*shape, generator=g, dtype=DT)


def choice(g, items):
    return items[min(len(items) - 1, int(uf(g, 0, len(items))))]


def vec(*xs):
    return torch.tensor(xs, dtype=DT)


# --------------------------------------------------------------------------- textures
def make_texture(g, kind, base=None):
    base = uv(g, 0.15, 0.85, 3) if base is None else base
    k = 10
    dirs = nv(g, k, 3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    tex = {
        "kind": kind,
        "base": base,
        "omega": dirs * uv(g, 3.0, 14.0, k)[:, None],
        "phase": uv(g, 0.0, 2 * math.pi, k),
        "amp": uv(g, 0.02, 0.07, k, 3),
    }
    if kind == "brick":
        tex.update(bh=uf(g, 0.18, 0.3), ratio=uf(g, 1.8, 2.6), mortar=0.025, mortar_col=uv(g, 0.55, 0.8, 3))
    elif kind == "tile":
        tex.update(ts=uf(g, 0.35, 0.7), grout=0.03, alt=uv(g, 0.7, 1.0, 3))
    elif kind == "stripes":
        sd = nv(g, 3)
        tex.update(sdir=sd / sd.norm(), sfreq=uf(g, 1.5, 4.0), alt=uv(g, 0.5, 0.9, 3))
    elif kind == "checker":
        tex.update(cs=uf(g, 0.15, 0.35), alt=uv(g, 0.45, 0.85, 3))
    return tex


def frac(x):
    return x - torch.floor(x)


def albedo(x, n, tex):
    wobble = torch.sin(x @ tex["omega"].T + tex["phase"]) @ tex["amp"]
    col = tex["base"][None, :] * (1.0 + wobble)
    kind = tex["kind"]
    if kind == "brick":
        hcoord = torch.where(n[:, 0].abs() > 0.5, x[:, 2], x[:, 0])
        bh = tex["bh"]
        bw = bh * tex["ratio"]
        m = tex["mortar"]
        row = torch.floor(x[:, 1] / bh)
        off = 0.5 * bw * torch.remainder(row, 2.0)
        col_idx = torch.floor((hcoord + off) / bw)
        fy = x[:, 1] / bh - row
        fx = (hcoord + off) / bw - col_idx
        mortar = (fy < m / bh) | (fx < m / bw)
        tint = 0.85 + 0.3 * frac(torch.sin(row * 12.9898 + col_idx * 78.233) * 43758.5453)
        col = torch.where(mortar[:, None], tex["mortar_col"][None, :].expand_as(col), col * tint[:, None])
    elif kind == "tile":
        ts = tex["ts"]
        a = x[:, 0] / ts
        b = x[:, 2] / ts
        ia, ib = torch.floor(a), torch.floor(b)
        grout = ((a - ia) < tex["grout"] / ts) | ((b - ib) < tex["grout"] / ts)
        checker = torch.remainder(ia + ib, 2.0) > 0.5
        col = torch.where(checker[:, None], col * tex["alt"][None, :], col)
        col = torch.where(grout[:, None], col * 0.45, col)
    elif kind == "stripes":
        t = torch.sin((x @ tex["sdir"]) * tex["sfreq"] * 2 * math.pi)
        f = 0.5 + 0.5 * torch.tanh(4.0 * t)
        col = col * (tex["alt"][None, :] + (1.0 - tex["alt"][None, :]) * f[:, None])
    elif kind == "checker":
        c = torch.floor(x / tex["cs"]).sum(dim=1)
        odd = torch.remainder(c, 2.0) > 0.5
        col = torch.where(odd[:, None], col * tex["alt"][None, :], col)
    return col.clamp(0.02, 0.95)


# --------------------------------------------------------------------------- geometry
def box(lo, hi, tex):
    return {"type": "box", "lo": vec(*lo), "hi": vec(*hi), "tex": tex}


def sphere(c, r, tex):
    return {"type": "sphere", "c": vec(*c), "r": float(r), "tex": tex}


def ray_box(o, d, lo, hi):
    d_safe = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    inv = 1.0 / d_safe
    t1 = (lo[None, :] - o) * inv
    t2 = (hi[None, :] - o) * inv
    tnear, axis = torch.minimum(t1, t2).max(dim=1)
    tfar = torch.maximum(t1, t2).min(dim=1).values
    hit = (tfar >= tnear) & (tfar > EPS)
    t = torch.where(tnear > EPS, tnear, tfar)
    t = torch.where(hit, t, torch.full_like(t, math.inf))
    n = torch.zeros_like(o)
    sgn = -torch.sign(torch.gather(d_safe, 1, axis[:, None]))
    n.scatter_(1, axis[:, None], sgn)
    return t, n


def ray_sphere(o, d, c, r):
    oc = o - c[None, :]
    b = (oc * d).sum(dim=1)
    cc = (oc * oc).sum(dim=1) - r * r
    disc = b * b - cc
    sq = torch.sqrt(disc.clamp_min(0.0))
    t0, t1 = -b - sq, -b + sq
    t = torch.where(t0 > EPS, t0, t1)
    hit = (disc >= 0) & (t > EPS)
    t = torch.where(hit, t, torch.full_like(t, math.inf))
    p = o + torch.where(hit, t, torch.zeros_like(t))[:, None] * d
    return t, (p - c[None, :]) / r


def prim_hit(p, o, d):
    if p["type"] == "box":
        return ray_box(o, d, p["lo"], p["hi"])
    return ray_sphere(o, d, p["c"], p["r"])


def intersect(prims, o, d):
    best = torch.full((o.shape[0],), math.inf, dtype=DT)
    pid = torch.full((o.shape[0],), -1, dtype=torch.long)
    nrm = torch.zeros_like(o)
    for k, p in enumerate(prims):
        t, n = prim_hit(p, o, d)
        closer = t < best
        best = torch.where(closer, t, best)
        pid = torch.where(closer, torch.full_like(pid, k), pid)
        nrm = torch.where(closer[:, None], n, nrm)
    return best, pid, nrm


def occluded(prims, o, d, tmax):
    occ = torch.zeros(o.shape[0], dtype=torch.bool)
    for p in prims:
        t, _ = prim_hit(p, o, d)
        occ |= t < tmax
    return occ


def inside_any(prims, x, margin):
    for p in prims:
        if p["type"] == "box":
            if bool(((x > p["lo"] - margin) & (x < p["hi"] + margin)).all()):
                return True
        elif float((x - p["c"]).norm()) < p["r"] + margin:
            return True
    return False


def look_at(center, target):
    f = target - center
    f = f / f.norm()
    up = vec(0.0, 1.0, 0.0)
    r = torch.linalg.cross(f, up)
    r = r / r.norm()
    dn = torch.linalg.cross(f, r)
    rot = torch.stack([r, dn, f], dim=0)  # rows: camera x (right), y (down), z (forward)
    return rot, -rot @ center


# --------------------------------------------------------------------------- scene layout
def build_layout(g):
    prims = []
    prims.append(box((-6.0, -0.2, -6.0), (6.0, 0.0, 6.0), make_texture(g, "tile")))
    prims.append(box((-6.0, 0.0, 5.0), (6.0, 5.0, 5.4), make_texture(g, "brick")))
    prims.append(box((-6.0, 0.0, -6.4), (6.0, 5.0, -6.0), make_texture(g, "brick")))
    side = make_texture(g, "brick")
    prims.append(box((-6.0, 0.0, -6.0), (-5.6, 5.0, 6.0), side))
    prims.append(box((5.6, 0.0, -6.0), (6.0, 5.0, 6.0), side))

    mw, mh, md = uf(g, 1.4, 2.4), uf(g, 1.2, 2.2), uf(g, 0.8, 1.4)
    mx, mz = uf(g, -0.6, 0.6), uf(g, 1.4, 2.4)
    main_tex = make_texture(g, choice(g, ["stripes", "checker", "noise"]))
    prims.append(box((mx - mw / 2, 0.0, mz - md / 2), (mx + mw / 2, mh, mz + md / 2), main_tex))
    tw, th = uf(g, 0.35, 0.7), uf(g, 0.5, 1.3)
    tx = uf(g, mx - mw / 2 + tw / 2, mx + mw / 2 - tw / 2)
    tz = uf(g, mz - md / 2 + tw / 2, mz + md / 2 - tw / 2)
    prims.append(box((tx - tw / 2, mh, tz - tw / 2), (tx + tw / 2, mh + th, tz + tw / 2),
                     make_texture(g, choice(g, ["stripes", "checker", "noise"]))))
    sh_, sd = uf(g, 0.12, 0.3), uf(g, 0.35, 0.6)
    prims.append(box((mx - 0.45 * mw, 0.0, mz - md / 2 - sd), (mx + 0.45 * mw, sh_, mz - md / 2),
                     make_texture(g, "noise")))

    props = []
    n_props = 3 + min(2, int(uf(g, 0, 3)))
    for _ in range(n_props):
        px = pz = None
        for _attempt in range(80):
            cx_, cz_ = uf(g, -2.8, 2.8), uf(g, -0.5, 3.8)
            if abs(cx_ - mx) < mw / 2 + 0.6 and abs(cz_ - mz) < md / 2 + sd + 0.6:
                continue
            if any((cx_ - ax) ** 2 + (cz_ - az) ** 2 < 0.9 ** 2 for ax, az in props):
                continue
            px, pz = cx_, cz_
            break
        if px is None:
            continue
        props.append((px, pz))
        tex = make_texture(g, choice(g, ["stripes", "checker", "noise"]))
        if uf(g, 0, 1) < 0.4:
            r = uf(g, 0.25, 0.55)
            if uf(g, 0, 1) < 0.5:
                ph = uf(g, 0.3, 0.8)
                prims.append(box((px - 0.45 * r, 0.0, pz - 0.45 * r), (px + 0.45 * r, ph, pz + 0.45 * r),
                                 make_texture(g, "noise")))
                prims.append(sphere((px, ph + r, pz), r, tex))
            else:
                prims.append(sphere((px, r, pz), r, tex))
        else:
            w = uf(g, 0.25, 0.5)
            h = uf(g, 0.7, 2.0)
            prims.append(box((px - w / 2, 0.0, pz - w / 2), (px + w / 2, h, pz + w / 2), tex))

    palette = [vec(1.0, 0.72, 0.42), vec(0.55, 0.75, 1.0), vec(1.0, 0.5, 0.8), vec(0.6, 1.0, 0.6)]
    lamps = []
    for _ in range(3):
        pos = None
        for _attempt in range(200):
            cand = vec(uf(g, -4.8, 4.8), uf(g, 1.0, 3.0), uf(g, -1.5, 4.6))
            if not inside_any(prims, cand, 0.15):
                pos = cand
                break
        if pos is None:
            continue
        lamps.append({"p": pos, "color": choice(g, palette) * uf(g, 0.9, 1.1), "r": uf(g, 0.9, 1.8)})
    look = vec(mx, 0.8, mz - 0.6)
    return prims, lamps, look


def make_cameras(g, prims, look):
    az = torch.linspace(-75.0, 75.0, N_VIEWS, dtype=DT) + uv(g, -1.5, 1.5, N_VIEWS)
    cams = []
    for i in range(N_VIEWS):
        a = math.radians(float(az[i]))
        center = None
        for _attempt in range(100):
            r, h = uf(g, 3.6, 4.8), uf(g, 0.9, 2.4)
            cand = vec(float(look[0]) + r * math.sin(a), h, float(look[2]) - r * math.cos(a))
            if not inside_any(prims, cand, 0.6):
                center = cand
                break
        if center is None:
            center = cand
        target = look + nv(g, 3) * 0.25
        rot, t = look_at(center, target)
        cams.append({"R": rot, "t": t, "C": center})
    return cams


# --------------------------------------------------------------------------- appearance
SESSION_TYPES = ["day", "golden", "overcast", "night"]


def sun_direction(elev_deg, az_deg):
    el, az = math.radians(elev_deg), math.radians(az_deg)
    return vec(math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az))


def make_sessions(g, n_lamps):
    types = ["day", "night", choice(g, SESSION_TYPES), choice(g, SESSION_TYPES)]
    sessions = []
    for ty in types:
        s = {"type": ty, "elev": 0.0, "az": uf(g, 0.0, 360.0), "sun": vec(0.0, 0.0, 0.0)}
        lamps = torch.zeros(n_lamps, dtype=DT)
        if ty == "day":
            s["elev"] = uf(g, 35.0, 65.0)
            s["sun"] = vec(1.0, 0.96, 0.9) * uf(g, 2.0, 3.0)
            s["amb"] = vec(0.35, 0.42, 0.55) * uf(g, 0.6, 1.0)
            s["sky"] = vec(0.5, 0.65, 0.9) * uf(g, 1.5, 2.5)
        elif ty == "golden":
            s["elev"] = uf(g, 8.0, 20.0)
            s["sun"] = vec(1.0, 0.62, 0.32) * uf(g, 1.5, 2.5)
            s["amb"] = vec(0.25, 0.2, 0.3) * uf(g, 0.6, 1.0)
            s["sky"] = vec(0.9, 0.55, 0.35) * uf(g, 1.0, 1.5)
            for k in range(n_lamps):
                if uf(g, 0, 1) < 0.5:
                    lamps[k] = uf(g, 0.3, 0.8)
        elif ty == "overcast":
            s["amb"] = vec(0.85, 0.88, 0.92) * uf(g, 0.8, 1.2)
            s["sky"] = vec(0.75, 0.77, 0.8) * uf(g, 1.0, 1.5)
        else:
            s["amb"] = vec(0.03, 0.04, 0.08) * uf(g, 0.6, 1.2)
            s["sky"] = vec(0.02, 0.03, 0.07)
            for k in range(n_lamps):
                if uf(g, 0, 1) < 0.8:
                    lamps[k] = uf(g, 1.5, 3.0)
            if n_lamps and float(lamps.max()) == 0.0:
                lamps[0] = uf(g, 1.5, 3.0)
        s["lamps"] = lamps
        sessions.append(s)
    return sessions


def make_capture(g, sess):
    elev = sess["elev"] + float(nv(g, 1)) * 1.5
    az = sess["az"] + float(nv(g, 1)) * 3.0
    return {
        "session": sess["type"],
        "sun_on": float(sess["sun"].max()) > 0,
        "sun_dir": sun_direction(max(elev, 3.0), az),
        "sun": sess["sun"] * math.exp(0.08 * float(nv(g, 1))),
        "amb": sess["amb"] * math.exp(0.08 * float(nv(g, 1))),
        "sky": sess["sky"] * math.exp(0.1 * float(nv(g, 1))),
        "lamps": sess["lamps"] * torch.exp(0.15 * nv(g, len(sess["lamps"]))),
        "wb": torch.exp(0.07 * nv(g, 3)),
        "ev": math.exp(0.3 * float(nv(g, 1))),
        "white": uf(g, 2.0, 4.0),
        "sat": uf(g, 0.75, 1.25),
        "black": uf(g, 0.0, 0.04),
        "k": None,
    }


# --------------------------------------------------------------------------- rendering
def trace_camera(prims, lamps, cam):
    hs, ws = H * SS, W * SS
    vv = (torch.arange(hs, dtype=DT) + 0.5) / SS - 0.5
    uu = (torch.arange(ws, dtype=DT) + 0.5) / SS - 0.5
    gv, gu = torch.meshgrid(vv, uu, indexing="ij")
    dc = torch.stack([(gu - CX) / FOCAL, (gv - CY) / FOCAL, torch.ones_like(gu)], dim=-1).reshape(-1, 3)
    d = dc @ cam["R"]
    d = d / d.norm(dim=1, keepdim=True)
    o = cam["C"][None, :].expand(d.shape[0], 3).contiguous()
    t, pid, n = intersect(prims, o, d)
    hit = torch.isfinite(t)
    tt = torch.where(hit, t, torch.zeros_like(t))
    x = o + tt[:, None] * d
    alb = torch.zeros_like(x)
    for k, p in enumerate(prims):
        m = pid == k
        if bool(m.any()):
            alb[m] = albedo(x[m], n[m], p["tex"])
    zdepth = tt * (d @ cam["R"][2])
    amb = 0.55 + 0.45 * n[:, 1]
    terms = []
    for lamp in lamps:
        lv = lamp["p"][None, :] - x
        dist = lv.norm(dim=1).clamp_min(1e-6)
        ldir = lv / dist[:, None]
        cos = (n * ldir).sum(dim=1).clamp_min(0.0)
        need = hit & (cos > 0)
        vis = torch.zeros_like(hit)
        if bool(need.any()):
            vis[need] = ~occluded(prims, x[need] + 1e-4 * n[need], ldir[need], dist[need])
        terms.append(cos * vis.to(DT) / (1.0 + (dist / lamp["r"]) ** 2))
    lamp_terms = torch.stack(terms, 0) if terms else torch.zeros(0, x.shape[0], dtype=DT)
    return {"hit": hit, "x": x, "n": n, "alb": alb, "z": zdepth, "amb": amb, "lamps": lamp_terms}


def sun_term(prims, geo, sun_dir):
    cos = (geo["n"] @ sun_dir).clamp_min(0.0)
    need = geo["hit"] & (cos > 0)
    vis = torch.zeros_like(geo["hit"])
    if bool(need.any()):
        m = int(need.sum())
        vis[need] = ~occluded(prims, geo["x"][need] + 1e-4 * geo["n"][need], sun_dir[None, :].expand(m, 3), math.inf)
    return cos * vis.to(DT)


def shade(geo, lamps, cap, sun_t):
    light = cap["amb"][None, :] * geo["amb"][:, None]
    if cap["sun_on"]:
        light = light + cap["sun"][None, :] * sun_t[:, None]
    if len(lamps):
        colors = torch.stack([lp["color"] for lp in lamps], 0)
        light = light + (geo["lamps"] * cap["lamps"][:, None]).T @ colors
    rad = geo["alb"] * light
    rad = torch.where(geo["hit"][:, None], rad, cap["sky"][None, :].expand_as(rad))
    img = rad.reshape(H * SS, W * SS, 3).permute(2, 0, 1)
    return F.avg_pool2d(img[None], SS)[0]


def exposure(lin, cap):
    x = lin * cap["wb"][:, None, None]
    lum = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
    return float(min(200.0, max(0.05, 0.18 * cap["ev"] / max(float(lum.mean()), 1e-4))))


def isp(lin, cap):
    x = lin * (cap["k"] * cap["wb"])[:, None, None]
    w2 = cap["white"] ** 2
    x = x * (1.0 + x / w2) / (1.0 + x)
    x = x.clamp(0.0, 1.0) ** (1.0 / 2.2)
    lum = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
    x = lum[None] + cap["sat"] * (x - lum[None])
    x = cap["black"] + (1.0 - cap["black"]) * x
    return x.clamp(0.0, 1.0)


def quantize(img):
    return (img * 255.0).round().clamp(0, 255).to(torch.uint8)


# --------------------------------------------------------------------------- scene assembly
def build_scene(sid, split, seed):
    t0 = time.time()
    g = torch.Generator().manual_seed(int(seed))
    prims, lamps, look = build_layout(g)
    cams = make_cameras(g, prims, look)
    sessions = make_sessions(g, len(lamps))
    perm = torch.randperm(N_VIEWS, generator=g).tolist()
    sess_of_view = [0] * N_VIEWS
    for rank, v in enumerate(perm):
        sess_of_view[v] = rank % len(sessions)
    caps = [make_capture(g, sessions[sess_of_view[v]]) for v in range(N_VIEWS)]
    train_idx = [v for v in range(N_VIEWS) if v not in HELDOUT_IDX]

    observed, zmaps, geos, sky_frac = {}, {}, {}, []
    for v in range(N_VIEWS):
        geo = trace_camera(prims, lamps, cams[v])
        sun_t = sun_term(prims, geo, caps[v]["sun_dir"]) if caps[v]["sun_on"] else None
        lin = shade(geo, lamps, caps[v], sun_t)
        caps[v]["k"] = exposure(lin, caps[v])
        observed[v] = quantize(isp(lin, caps[v]))
        zmaps[v] = geo["z"].reshape(H * SS, W * SS)
        sky_frac.append(1.0 - float(geo["hit"].to(DT).mean()))
        if v in HELDOUT_IDX:
            geos[v] = geo
        else:
            geos[v] = {"hit": geo["hit"], "x": geo["x"]}

    # Average appearance of each held-out camera over every training capture condition.
    avg = {}
    for h in HELDOUT_IDX:
        acc = torch.zeros(3, H, W, dtype=DT)
        for v in train_idx:
            sun_t = sun_term(prims, geos[h], caps[v]["sun_dir"]) if caps[v]["sun_on"] else None
            acc += isp(shade(geos[h], lamps, caps[v], sun_t), caps[v])
        avg[h] = acc / len(train_idx)

    # Sparse SfM-like point cloud: surface samples seen by the training views,
    # coloured by the mean observed colour over the training views that see them.
    per_view = int(math.ceil(N_INIT / len(train_idx)))
    pts = []
    for v in train_idx:
        hits = torch.nonzero(geos[v]["hit"]).squeeze(1)
        sel = hits[torch.randperm(hits.numel(), generator=g)[:per_view]]
        pts.append(geos[v]["x"][sel])
    pts = torch.cat(pts, 0)[:N_INIT]
    col_sum = torch.zeros(pts.shape[0], 3, dtype=DT)
    col_cnt = torch.zeros(pts.shape[0], dtype=DT)
    for v in train_idx:
        pc = (pts - cams[v]["C"][None, :]) @ cams[v]["R"].T
        z = pc[:, 2]
        zs = z.clamp_min(1e-6)
        u = FOCAL * pc[:, 0] / zs + CX
        w = FOCAL * pc[:, 1] / zs + CY
        j, i = torch.round(u).long(), torch.round(w).long()
        inb = (z > 0.05) & (j >= 0) & (j < W) & (i >= 0) & (i < H)
        js = torch.round((u + 0.5) * SS - 0.5).long().clamp(0, W * SS - 1)
        is_ = torch.round((w + 0.5) * SS - 0.5).long().clamp(0, H * SS - 1)
        zm = zmaps[v][is_, js]
        vis = inb & ((zm - z).abs() < 0.02 * z + 0.02)
        if bool(vis.any()):
            img = observed[v].to(DT) / 255.0
            col_sum[vis] += img[:, i[vis], j[vis]].T
            col_cnt[vis] += 1
    colors = torch.where(col_cnt[:, None] > 0, col_sum / col_cnt.clamp_min(1)[:, None], torch.full_like(col_sum, 0.5))
    pts_noisy = pts + 0.02 * nv(g, *pts.shape)

    def pack(ids):
        rs = torch.stack([cams[v]["R"] for v in ids]).float()
        ts = torch.stack([cams[v]["t"] for v in ids]).float()
        k = torch.tensor([[FOCAL, 0.0, CX], [0.0, FOCAL, CY], [0.0, 0.0, 1.0]], dtype=torch.float32)
        return rs, ts, k[None].repeat(len(ids), 1, 1)

    tr_r, tr_t, tr_k = pack(train_idx)
    ho_r, ho_t, ho_k = pack(HELDOUT_IDX)
    visible = {
        "scene_id": sid,
        "split": split,
        "height": H,
        "width": W,
        "train_images": torch.stack([observed[v] for v in train_idx]),
        "train_R": tr_r,
        "train_t": tr_t,
        "train_K": tr_k,
        "heldout_left": torch.stack([observed[h][:, :, : W // 2] for h in HELDOUT_IDX]).contiguous(),
        "heldout_R": ho_r,
        "heldout_t": ho_t,
        "heldout_K": ho_k,
        "points": pts_noisy.float(),
        "colors": colors.float(),
    }
    answers = {
        "right": torch.stack([observed[h][:, :, W // 2:] for h in HELDOUT_IDX]).contiguous(),
        "avg": torch.stack([avg[h] for h in HELDOUT_IDX]).to(torch.float16),
    }
    means = [float(observed[v].float().mean() / 255.0) for v in range(N_VIEWS)]
    print(
        f"[build] {sid} ({split}) prims={len(prims)} lamps={len(lamps)} "
        f"sessions={[s['type'] for s in sessions]} sky_frac_max={max(sky_frac):.3f} "
        f"mean_intensity=[{min(means):.2f},{max(means):.2f}] points={pts.shape[0]} "
        f"time={time.time() - t0:.1f}s",
        flush=True,
    )
    return visible, answers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the manifest exists")
    args = ap.parse_args()
    manifest = DATA_DIR / "manifest.json"
    if manifest.exists() and TRAIN_ANSWERS.exists() and TEST_ANSWERS.exists() and not args.force:
        print(f"[build] benchmark already present ({manifest}); use --force to rebuild")
        return
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries, train_ans, test_ans = [], {}, {}
    for split, seeds in (("dev", DEV_SEEDS), ("test", TEST_SEEDS)):
        for sid, seed in seeds.items():
            visible, answers = build_scene(sid, split, seed)
            torch.save(visible, DATA_DIR / f"{sid}.pt")
            (train_ans if split == "dev" else test_ans)[sid] = answers
            entries.append({"id": sid, "split": split, "file": f"{sid}.pt"})
    meta = {"version": 1, "height": H, "width": W, "num_heldout": len(HELDOUT_IDX)}
    torch.save({**meta, "scenes": train_ans}, TRAIN_ANSWERS)
    torch.save({**meta, "scenes": test_ans}, TEST_ANSWERS)
    manifest.write_text(json.dumps({**meta, "num_train": N_VIEWS - len(HELDOUT_IDX), "scenes": entries}, indent=2) + "\n")
    print(f"[build] wrote {len(entries)} scenes to {DATA_DIR}")


if __name__ == "__main__":
    main()
