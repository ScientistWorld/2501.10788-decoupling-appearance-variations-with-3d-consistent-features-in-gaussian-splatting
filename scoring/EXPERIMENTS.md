# Experiment Overview

## glav_novel_view (weight 0.7)

This is the central question. After reconstructing a scene from photos whose appearance varies globally (camera processing, time of day) and locally (lamps, moving shadows), can a method render a held-out capture accurately? It may infer that capture's appearance only from the capture's left half, and it is scored on the right half with PSNR, SSIM and LPIPS. The test averages 36 held-out captures over six hidden scenes, so a gain must hold across scenes and capture conditions to count.

## glav_average_appearance (weight 0.3)

This checks the other side of the claim. Better held-out renders must not come at the cost of a corrupted scene. The scene rendered without any per-capture appearance state is compared with the camera's average appearance over the training captures, after a global colour correction owned by the evaluator, on the same held-out right halves. Floaters, baked-in lighting and colour distortions lower the score. It is also a hard constraint. A run whose appearance-free scene is clearly worse than a plain reconstruction (by more than 1 dB PSNR) is scored as failed in both experiments. The two degraded comparison variants show how much this experiment moves when a component of a full system is removed.
