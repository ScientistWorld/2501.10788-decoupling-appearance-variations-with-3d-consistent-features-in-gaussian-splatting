# Decoupling Appearance Variations with 3D Consistent Features in Gaussian Splatting

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2501.10788"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2501.10788-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2501.10788](https://arxiv.org/abs/2501.10788) |
|:---|:---|
| **Domain** | Computer Vision / Novel View Synthesis |
| **TL;DR** | DAVIGS improves Gaussian Splatting under in-the-wild appearance variation by learning image-level affine color transforms conditioned on per-view embeddings and 3D-consistent hash-grid features. |

## Summary

The paper addresses floaters and color distortions caused by inconsistent appearance across multi-view captures. DAVIGS keeps the appearance model decoupled from Gaussian attributes by transforming rendered images rather than persistent Gaussian properties. It uses depth back-projection, multi-resolution hash-grid features, per-view appearance embeddings, an MLP-decoded affine color map, identity regularization, and cell-based querying. Reported results show better quality than vanilla 3DGS and other appearance-aware baselines while preserving high rendering speed.

## Problem

Novel-view synthesis systems built from casually captured photos assume that the same 3D point has consistent color across views. Real captures often violate that assumption because camera processing, time of day, weather, and local illumination vary between images. The task is to reconstruct and render a scene despite those per-image appearance changes, without leaving floaters or color artifacts in the average scene representation.

## The paper's method

DAVIGS is a decoupled appearance modeling method for Gaussian Splatting. It models image-specific global and local appearance variation during optimization, but transforms rendered images at the image level so the appearance module can be discarded for final rendering.

## Key results of the paper

- On GLAV, the paper reports DAVIGS improving PSNR from 22.80 for vanilla 3DGS to 28.31.
- On PhotoTourism, DAVIGS reports comparable quality to coupled methods with lower optimization memory and no persistent rendering slowdown.
- Ablations show that replacing 3D features with 2D/depth/color features or removing identity regularization degrades quality.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | +163.0 |
| `glm-5.2-xhigh` | +139.5 |
| `gpt-5.6-terra-xhigh` | +112.1 |
| `gpt-5.5-xhigh` | +100.1 |
| `MiniMax-M3-xhigh` | +14.5 |
| `gpt-5.6-luna-xhigh` | -5.7 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2501.10788 --mode innovate --agent claude-code
sw evaluate 2501.10788
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

