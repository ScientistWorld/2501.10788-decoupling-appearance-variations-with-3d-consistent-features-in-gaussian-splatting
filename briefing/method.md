# Paper's Method

## Key Contribution

DAVIGS is a decoupled appearance modeling method for Gaussian Splatting. It models image-specific global and local appearance variation during optimization, but transforms rendered images at the image level so the appearance module can be discarded for final rendering.

## Approach

For each training view, Gaussian Splatting renders both an RGB image and a depth map. Each rendered pixel is back-projected into 3D using depth and camera parameters. DAVIGS queries multi-resolution hash grids at those 3D locations to obtain local 3D-consistent features, concatenates them with a learned per-view appearance embedding, and decodes a 3x4 affine color transform with a lightweight MLP.

The affine transform is applied to the rendered pixel color before computing the photometric loss against the captured image. An identity regularizer keeps the affine maps close to the identity matrix so scene colors remain stable instead of letting the appearance module absorb arbitrary reconstruction errors. For efficiency, the paper queries transforms on a coarse cell grid and bilinearly interpolates the affine matrices to pixels.

## Main Claims

The primary claim is that 3D-consistent image-level appearance modeling improves rendering quality on scenes with global and local appearance variations while preserving fast rendering because the appearance module is decoupled from final rendering.

Secondary claims are that the method adds little optimization memory/time overhead, works as a plug-in module for several Gaussian Splatting variants, and that 3D features plus identity regularization are both important.

## Reproduction in this workspace (`method/`)

- **Foundation:** 3D Gaussian Splatting in plain PyTorch (`method/gaussians.py`) with anisotropic covariances, SH degree 3 (raised progressively), EWA projection, depth-sorted alpha compositing, rendered mean depth, clone/split/prune densification with opacity resets, and the 3DGS per-parameter learning rates. Everything is scaled to 6,000 iterations: densify from 1/60 to 50% of training every 1/300 of training, reset opacity every 10%, and cap the model at 6,000 Gaussians.
- **DAVIGS** (`method/appearance.py`): a 16-level x 2-feature hash grid (2^15 entries per level, resolutions 16 to 128) over the point-cloud bounding box, 32-d per-capture embeddings, an MLP with hidden widths 128 and 64 that outputs a 3x4 affine map (initialised to identity), and the cell-based query with cell size 1 px. The paper's 8 px cells at 1080p are 0.74% of the image height, less than one pixel at 96 px, and the paper's Table 5 shows cell 1 and cell 8 give the same quality.
- **Unspecified hyperparameters:** the paper does not give the hash grid's finest resolution or the module learning rates. Four candidates (finest resolution 512 or 128, module learning rates x1 or x0.2) were trained on the three development scenes and scored with the public development-scene scorer (`method/select_config.py`). Resolution 128 at x1 learning rates won on every novel-view metric and on the appearance-free scene; the choice is stored in `method/selected_config.json`.
- **Loss:** (1-λ1)·L1 + λ1·D-SSIM with λ1 = 0.2, plus λ2·L_ID. λ2 warms up linearly from 0 to 0.3 over the first sixth of training (the paper's 5k of 30k) and then cosine-anneals to 0.2.
- **Test protocol:** the held-out captures' left halves are used during training, interleaved after every 5 training steps, to optimise only that capture's appearance embedding. Neither the Gaussians nor the shared module receive gradient from them.
- **Outputs:** the transformed held-out renders (right halves) and the Gaussians themselves; the evaluator renders the Gaussians without the appearance module.
- **Ablations:** `davigs_uv_depth` replaces the hash features with the raw values and 5-frequency sin/cos encodings of (u, v, depth); `davigs_no_id` sets λ2 = 0.
