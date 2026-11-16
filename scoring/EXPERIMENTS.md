# Experiment Overview

## glav_novel_view (weight 0.7)

This is the central question. After reconstructing a scene from photos whose appearance varies globally (camera processing, time of day) and locally (lamps, moving shadows), can a method render a held-out capture accurately? It may infer that capture's appearance only from the capture's left half, and it is scored on the right half with PSNR, SSIM and LPIPS. The test averages 36 held-out captures over six hidden scenes, so a gain must hold across scenes and capture conditions to count. The plain reconstruction and two degraded variants of a full system are fixed comparison points.

## glav_average_appearance (weight 0.3)

This checks the other side of the claim. Better held-out renders must not come at the cost of a corrupted scene. Each method hands in its reconstructed 3D Gaussian scene; the evaluator renders it itself, with no appearance model or per-capture state, from six viewpoints per scene where no photo was ever taken. Each render is compared with the camera's average appearance over the training captures after a global colour correction owned by the evaluator. Floaters, baked-in lighting, holes and colour distortions lower the score, and a global colour or exposure offset does not. It is also a hard constraint: a run whose scene renders more than 1 dB PSNR worse here than the evaluator's frozen reference reconstruction is scored as failed in both experiments. The degraded comparison variants show how much this experiment moves when a component of a full system is removed.
