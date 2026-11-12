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

- **Foundation:** 3D Gaussian Splatting in plain PyTorch (`method/gaussians.py`) with anisotropic covariances, SH degree 3 (raised progressively), EWA projection, depth-sorted alpha compositing, rendered mean depth, clone/split/prune densification with opacity resets, and the 3DGS per-parameter learning rates. Everything is scaled to 6,000 iterations: densify from 5% to 50% of training every 1/60 of training, reset opacity every 20%, and cap the model at 6,000 Gaussians.
- **DAVIGS** (`method/appearance.py`): a 16-level × 2-feature hash grid (2^15 entries per level, resolutions 16 to 512) over the point-cloud bounding box, 32-d per-capture embeddings, an MLP with hidden widths 128 and 64 that outputs a 3×4 affine map (initialised to identity), and a cell-based query with bilinear interpolation of the matrices (cell size 2 px at 96×128, which is finer than the paper's 8 px at 1080p in absolute terms but coarser relative to the image).
- **Loss:** (1−λ1)·L1 + λ1·D-SSIM with λ1 = 0.2, plus λ2·L_ID. λ2 warms up linearly from 0 to 0.3 over the first sixth of training (the paper's 5k of 30k) and then cosine-anneals to 0.2.
- **Test protocol:** the held-out captures' left halves are used during training, interleaved after every 5 training steps, to optimise appearance-related parameters only (embedding, hash grid, MLP). Gaussians receive no gradient from them.
- **Ablations:** `davigs_uv_depth` replaces the hash features with the raw values and 5-frequency sin/cos encodings of (u, v, depth); `davigs_no_id` sets λ2 = 0.
