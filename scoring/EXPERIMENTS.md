# Experiment Overview

## synthetic_glav_core

This experiment tests the central benchmark question: whether a scene reconstruction system can render held-out views accurately when training and test images contain both image-wide and spatially local appearance changes. It also checks that the result remains usable as a decoupled renderer rather than depending on additional appearance parameters at final render time.

## synthetic_glav_ablation

This experiment checks that performance comes from the intended component choices rather than just adding any image-level correction. It compares the full method with variants that remove spatially consistent scene information or remove the stabilizing transformation constraint, using the same held-out view protocol.
