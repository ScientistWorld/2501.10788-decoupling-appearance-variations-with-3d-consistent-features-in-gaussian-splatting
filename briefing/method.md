# Paper's Method

## Key Contribution

DAVIGS is a decoupled appearance modeling method for Gaussian Splatting. It models image-specific global and local appearance variation during optimization, but transforms rendered images at the image level so the appearance module can be discarded for final rendering.

## Approach

For each training view, Gaussian Splatting renders both an RGB image and a depth map. Each rendered pixel is back-projected into 3D using depth and camera parameters. DAVIGS queries multi-resolution hash grids at those 3D locations to obtain local 3D-consistent features, concatenates them with a learned per-view appearance embedding, and decodes a 3x4 affine color transform with a lightweight MLP.

The affine transform is applied to the rendered pixel color before computing the photometric loss against the captured image. An identity regularizer keeps the affine maps close to the identity matrix so scene colors remain stable instead of letting the appearance module absorb arbitrary reconstruction errors. For efficiency, the paper queries transforms on a coarse cell grid and bilinearly interpolates the affine matrices to pixels.

## Main Claims

The primary claim is that 3D-consistent image-level appearance modeling improves rendering quality on scenes with global and local appearance variations while preserving fast rendering because the appearance module is decoupled from final rendering.

Secondary claims are that the method adds little optimization memory/time overhead, works as a plug-in module for several Gaussian Splatting variants, and that 3D features plus identity regularization are both important.
