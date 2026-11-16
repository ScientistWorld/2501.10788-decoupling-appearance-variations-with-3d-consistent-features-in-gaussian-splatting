# Problem Definition

## Research Question

Novel-view synthesis systems built from casually captured photos assume that the same 3D point has consistent colour across views. Real captures violate that assumption: camera processing (exposure, white balance, tone curves), time of day, weather, and local illumination (street lamps, moving cast shadows) change between images. The task is to reconstruct a scene from such a collection and

1. render a held-out capture accurately once its appearance has been inferred from part of that capture, and
2. keep the reconstructed scene itself clean (no floaters or colour distortions) so it can be rendered without any appearance model or per-capture state.

## Benchmark Data

`data/glav_syn/` holds nine scenes (`dev_00..02` for development, `test_00..05` for the held-out evaluation). Each is a walled courtyard with textured buildings and props, captured by 36 cameras on an arc, under four capture sessions per scene (day, golden hour, overcast, night) with per-capture camera-processing changes. Lamps and the sun produce spatially local changes that are consistent in 3D across captures of the same session.

Per scene a method receives (see `data/glav_data.py`):

- 30 training captures (96x128 RGB, uint8) with their cameras (world-to-camera `R`, `t`, intrinsics `K`, OpenCV convention),
- 6 held-out cameras and the **left half** of each held-out capture,
- 6 canonical-only cameras, where no capture exists and the scene is rendered without appearance state,
- a sparse, noisy point cloud with colours, like a structure-from-motion output.

The right halves of the held-out captures and the evaluation targets are not available to methods.

## Why It Matters

Robust appearance handling makes 3D reconstruction usable for outdoor and in-the-wild captures where strict photometric consistency is unrealistic, while preserving fast, state-free rendering for downstream use.

## Success Criteria

Better held-out right-half quality (PSNR, SSIM, LPIPS) under global and local appearance variation, while the reconstructed Gaussian scene, rendered by the evaluator with no appearance model, stays clean.
