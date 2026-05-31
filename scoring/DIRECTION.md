# Research Direction

## Research Problem

This benchmark is about novel-view synthesis under inconsistent image appearance. The research question is how to reconstruct a stable 3D scene when training views contain global and local color/illumination changes that violate strict photometric consistency.

## Core Contribution

The relevant contribution family is appearance variation modeling for Gaussian Splatting-style scene representations. Improvements should focus on separating scene appearance from per-image capture variation while preserving 3D consistency and decoupled final rendering.

## Approach Scope

Scientist agents may redesign the appearance conditioning, feature representation, regularization, optimization schedule, or how test-view appearance is adapted. They may also improve the Gaussian splat optimizer as long as the task remains appearance-robust Gaussian-based reconstruction under the fixed benchmark protocol.

## Out of Bounds

Replacing the task with a 2D image-to-image translation system, using a non-Gaussian renderer as the main representation, training directly on the held-out right halves, adding external supervision, or requiring a persistent heavy rendering-time network is out of scope for this paper's claim.
