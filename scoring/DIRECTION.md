# Research Direction

## Research Problem

Novel-view synthesis under inconsistent image appearance. How should a stable 3D scene be reconstructed when training captures contain global and local colour/illumination changes that violate photometric consistency, so that held-out captures can be rendered once their appearance is inferred, while the scene itself stays clean?

## Core Contribution

The relevant contribution family is appearance-variation modelling for Gaussian-splatting scene representations. Improvements should better separate scene content from per-capture variation, preserve 3D consistency across captures, and keep final rendering free of per-capture appearance state.

## Approach Scope

Scientist agents may redesign the appearance conditioning, feature representation, regularisation, test-time appearance inference, the optimisation schedule and density control, as long as the scene is a Gaussian-based representation optimised on the provided captures.

## Out of Bounds

The following are out of scope:

- replacing the task with 2D image-to-image translation or copying training images;
- training scene geometry or colour on held-out left halves;
- submitting anything other than an appearance-free render as `average_appearance`;
- using a non-Gaussian renderer as the main representation;
- adding external data;
- requiring a persistent per-capture network at final render time.
