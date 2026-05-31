# Experiment Overview

## synthetic_glav_core

This experiment tests the central benchmark question: whether a scene reconstruction system can render held-out views accurately when training and test images contain both image-wide and spatially local appearance changes. It also checks that final rendering does not depend on persistent per-view correction state.

## synthetic_glav_ablation

This experiment checks whether the benchmark can distinguish a complete reconstruction system from controlled degraded variants under the same held-out view protocol. The goal is to verify that the score responds to meaningful modeling choices rather than any generic image-level correction.
