# Overview

- **Paper ID:** 2501.10788
- **Title:** Decoupling Appearance Variations with 3D Consistent Features in Gaussian Splatting
- **Domain:** Computer Vision / Novel View Synthesis
- **TL;DR:** DAVIGS improves Gaussian Splatting under in-the-wild appearance variation by learning image-level affine color transforms conditioned on per-view embeddings and 3D-consistent hash-grid features.

## Short Summary

The paper addresses floaters and color distortions caused by inconsistent appearance across multi-view captures. DAVIGS keeps the appearance model decoupled from Gaussian attributes by transforming rendered images rather than persistent Gaussian properties. It uses depth back-projection, multi-resolution hash-grid features, per-view appearance embeddings, an MLP-decoded affine color map, identity regularization, and cell-based querying. Reported results show better quality than vanilla 3DGS and other appearance-aware baselines while preserving high rendering speed.

## Key Results

- On GLAV, the paper reports DAVIGS improving PSNR from 22.80 for vanilla 3DGS to 28.31.
- On PhotoTourism, DAVIGS reports comparable quality to coupled methods with lower optimization memory and no persistent rendering slowdown.
- Ablations show that replacing 3D features with 2D/depth/color features or removing identity regularization degrades quality.
