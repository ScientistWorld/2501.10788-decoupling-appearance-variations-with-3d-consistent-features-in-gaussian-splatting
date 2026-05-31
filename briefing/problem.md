# Problem Definition

## Research Question

Novel-view synthesis systems built from casually captured photos assume that the same 3D point has consistent color across views. Real captures often violate that assumption because camera processing, time of day, weather, and local illumination vary between images. The task is to reconstruct and render a scene despite those per-image appearance changes, without leaving floaters or color artifacts in the average scene representation.

## Why It Matters

Robust appearance handling makes 3D reconstruction usable for outdoor and in-the-wild captures where strict photometric consistency is unrealistic, while preserving fast rendering for downstream visualization.

## Success Criteria

A successful solution should improve held-out rendering quality under global and local appearance variation, preserve multi-view 3D consistency, and avoid adding persistent rendering-time overhead once reconstruction is complete.
