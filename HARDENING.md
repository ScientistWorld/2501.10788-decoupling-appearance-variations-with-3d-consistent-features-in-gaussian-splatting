# Hardening report (iteration 11)

No `REVIEW.md` existed. This pass started from the state the earlier hardening attempt left in this iteration, whose 6-hour job had been killed. That state already had a hidden ray-traced generator, evaluator-owned targets and roster, worst values on failure, and a 3DGS + DAVIGS reproduction. All numbers below come from the unmodified hidden-test evaluator (`python -m eval.test.evaluate`): 6 test scenes, 36 held-out captures and 36 canonical views. The evidence is in `redteam/hardening/` (`honest_runs.json`, `shortcut_runs.json`, `summary.json`, `summary.md`) and in job `c2ce0afc-328`.

**Rank (dR).** dR applies the gym's rule: sum over metrics of experiment weight × |coefficient| × improvement in the metric's better direction ÷ SD. The final weights are 0.85 / 0.15; the "after" numbers below use them. dR is reported under two normalisations:

- **narrow:** SD across the reference's own seeds and the honest variants, which is pessimistic because the spreads are small;
- **broad:** the narrow cohort plus the three fixed comparison methods, which is closer to a diverse field of attempts.

## Does a better result now score higher?

**Yes.** Seed noise is measured with reference seeds 0, 1 and 2. Their dR is 0 / −0.16 / −0.90 (narrow) and 0 / −0.18 / −0.73 (broad). Their novel-view PSNR is 22.87 / 22.74 / 22.24 (range 0.62 dB).

| Variant (paired, same seeds/inputs) | NV PSNR | NV SSIM | NV LPIPS | AA PSNR | AA SSIM | AA LPIPS | dR narrow | dR broad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference, seed 0 | 22.866 | 0.7294 | 0.1579 | 26.274 | 0.7372 | 0.2219 | 0 | 0 |
| **better:** 1.5× optimisation steps (`long`, s0) | 23.548 | 0.7449 | 0.1459 | 27.584 | 0.7792 | 0.1996 | **+1.83** | **+1.33** |
| **better on the main experiment:** cosine-decayed module LR + 200 extra held-out embedding steps (`module_cosine`, s0/s1/s2) | 24.11 / 23.97 / 23.87 | 0.741 / 0.739 / 0.736 | 0.148 / 0.150 / 0.157 | 25.71 / 25.34 / 25.71 | 0.704 / 0.704 / 0.715 | 0.231 / 0.242 / 0.233 | +1.14 / +0.77 / +0.58 | +1.00 / +0.75 / +0.55 |
| **weaker:** no held-out appearance inference (`noadapt`, s0) | 19.646 | 0.6976 | 0.1903 | 26.215 | 0.7357 | 0.2207 | −3.98 | −3.20 |
| **weaker:** 0.25× steps (`short`, s0) | fails the gate (unscored AA 23.56 < 25.27; NV 19.89) → worst values | | | | | | −108.9 | −85.3 |

- **`long`** is better on all six metrics. It uses 1.5x the iteration budget, which `scoring/CONSTRAINTS.md` now fixes at the reference's 6,000. So it shows the test can resolve a genuinely better result, not an allowed route to a higher rank. Per-view paired t-statistics over 36 views: NV PSNR +0.68 dB (t = 3.1), NV SSIM t = 3.6, NV LPIPS t = −2.4; AA PSNR +1.31 dB (t = 5.9). Its dR exceeds the reference's whole seed spread.
- **`module_cosine`** keeps the reference's budget and representation. Paired by seed, it gains +1.25 / +1.23 / +1.63 dB NV PSNR (mean over views +1.37 dB, t = 7.6). It gives up 0.56–1.30 dB on the appearance-free scene. Paired dR differences are +1.14, +0.93 and +1.48, all above the seed spread.
- **Weaker variants** score lower.

**Before.**
- In the original gym (REDTEAM.md), the cosine variant's +0.40 ΔR at seed 7 turned into −0.22 at seed 8. The reference's three-seed ΔR range was 0.65, so separation was "not demonstrated".
- After the earlier hardening rewrite, the reference itself failed its own gate. Its appearance-free PSNR was 25.01 against a threshold of vanilla − 1 = 27.55. So the reference and every honest variant (seeds, cosine, cell-1, long, short, no-adapt) scored the worst value everywhere, and the test could not rank anything.

## Reference score before and after

| | NV PSNR | NV SSIM | NV LPIPS | AA PSNR | AA SSIM | AA LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Original gym (REDTEAM, different 2-view synthetic test, proxy perceptual metric) | 23.81 | 0.664 | 0.028 (proxy) | – | – | – |
| Before this pass (earlier rewrite, job `e6a83284`): **failed own gate → scored PSNR 0 / SSIM −1 / LPIPS 1**; unscored values → | 21.20 | 0.622 | 0.320 | 25.01 (old protocol) | 0.645 | 0.243 |
| **After** (`scoring/scores_test.json`) | **22.866** | **0.7294** | **0.1579** | **26.274** | **0.7372** | **0.2219** |
| vanilla 3DGS (frozen, after) | 18.704 | 0.7140 | 0.2086 | 27.367 | 0.7967 | 0.1474 |
| uv+depth ablation (frozen, after) | 20.863 | 0.7506 | 0.1747 | 28.112 | 0.8244 | 0.1239 |
| no-L_ID ablation (frozen, after) | 22.783 | 0.7068 | 0.1725 | 24.251 | 0.6466 | 0.2531 |

**Paper direction.**
- DAVIGS beats vanilla 3DGS on all three novel-view metrics (paper Table 1).
- It beats the no-L_ID ablation on all six metrics (paper Table 4).
- Against the uv+depth ablation it wins novel-view PSNR and LPIPS but loses novel-view SSIM and all three appearance-free metrics. The paper's Table 4 says the full model wins.
- Its appearance-free scene is also below vanilla's. The paper's "fewer floaters" claim is not reproduced quantitatively at this scale; see "Not fixed".

On the development scenes (`scores_train.json`, 3 scenes) the reference beats vanilla on novel-view PSNR (20.42 vs 17.46) and LPIPS. It trails vanilla on SSIM (0.588 vs 0.599) and the no-L_ID ablation on PSNR (20.91). Three scenes are noisy.

## What changed (all applied together, checked with one full job after two smoke runs)

1. **Reproduction restored toward the paper's settings.**
   - Cell size 2 → 1 px. The paper's 8 px cells at 1080p are 0.74% of the image height, below one pixel at 96 px, and its Table 5 shows cell 1 ≡ cell 8. In the earlier rewrite's 3-seed audit, cell 1 was better than cell 2 on all six metrics.
   - The two module hyperparameters the paper does not give (hash grid finest resolution, module learning rates) are now chosen on the development scenes by `method/select_config.py` from four candidates. 128 / ×1 won on every novel-view metric (dev criterion 30.8 vs 28.5 for the old 512 / ×1). At 512 the under-sampled finest levels caused per-pixel speckle in the transformed images.
   - Novel-view PSNR went from 21.20 to 22.87, and the reference now beats vanilla on SSIM and LPIPS, not only PSNR.
2. **The appearance-free experiment can no longer be gamed with images.**
   - **(a)** It is scored at six *canonical-only cameras* per scene where no capture exists. Previously it used the held-out cameras, where the adapted render could be submitted.
   - **(b)** Methods submit their **3D Gaussian scene**, and the evaluator renders it itself with its own 3DGS rasteriser. No appearance module or per-capture state can enter, and the cameras come from the evaluator-owned answers. The evaluator's render of the reference scene matches the method's own render at 100 dB PSNR (`_renderer_consistency`).
   - Scene validation fails runs for non-finite values, opacity outside [0, 1], non-positive scales, zero quaternions, more than 60,000 Gaussians, or a pathological render cost.
3. **Gate relative to the frozen reference reconstruction.** A method fails if its appearance-free PSNR is more than 1 dB below the frozen DAVIGS reference. Previously the gate was vanilla − 1 dB, which the reproduced method could not pass. The gate reference is evaluator-owned (`eval/*/frozen/davigs_reference.pt`), so the reference passes by construction (its seeds span 0.37 dB).
4. **Ablations scored in both experiments** as fixed comparison points.
5. **Weights 0.7/0.3 → 0.85/0.15** (`scoring/reference.json`).
   - **Why:** the paper's quantitative claims are on the novel-view protocol (Tables 1–3, the headline +5.5 dB). The colour-corrected protocol appears only for the Table 4 ablations, and the decoupling constraint is enforced by the gate.
   - **Effect:** at 0.3 the uv+depth ablation, 2.0 dB worse on the main metric, outranked the full method by +1.71 dR (narrow). At 0.15 it is +0.20 (narrow) / −0.21 (broad), within the reference's seed spread.
6. **Evaluator speed and the audit.**
   - Evaluator threads are capped. The honest variants are scored in one evaluator call (identical inputs). Audit training stops at a deadline so the audit is always scored. Stale or foreign-configuration runs are excluded from the audit.
   - Evaluating everything took 7 minutes, against more than an hour before.

## Shortcuts: fix and score before → after

| Shortcut | Before | Fix | After (dR narrow / broad) |
|---|---|---|---|
| Rebuild generator answers / brute-force generator seed | Pareto-perfect, ΔR +4.73 (REDTEAM) | Generator and 62-bit test seeds live only in `eval/test/` (earlier rewrite) | Not runnable from method-visible files |
| Replace targets inside the prediction file | PSNR 99, ΔR +4.73 | Targets are evaluator-owned; extra keys are ignored | `replace_targets`: extra keys ignored, scored on real targets: −53.3 / −43.4 |
| Replace the baseline / gate reference under `outputs/` | ΔR +9.82 | Comparisons and gate reference come from `eval/*/frozen/` only | `replace_baseline_and_gate`: 0.000 (exact reference copy) |
| Impossible or default metadata (iterations, coupling flag) | +1.05 / +0.04 | Nothing but images and the Gaussian scene is read | `metadata_injection`: 0.000 |
| Hide a failed method / unlisted method | tie (losses skipped) | Every method directory on disk plus the required `davigs` is scored; missing → worst values | `hide_failed_method`: worst (−108.9); `unlisted_method` scored |
| Reduce test size | +0.35 | Shapes and scene set checked against the answers | worst (−108.9) |
| NaN / ∞ / empty / missing / out-of-range | NaN accepted; empty/missing aborted with no score | All rejected → worst value in every metric | all worst (−108.9) |
| Submit the adapted render as the appearance-free render | +1.4 dB AA at smoke scale; +1.0 dB on test_03 at full scale (held-out-camera protocol) | Canonical-only cameras, then evaluator-rendered scene | not expressible (`images_instead_of_scene` fails) |
| Render the scene through the appearance module (coupled render, mean embedding) | +8.2 dR at smoke scale | The evaluator renders the submitted Gaussians itself | `canonical_via_module_images`: 0.000 (images ignored) |
| 2D baselines: nearest training image / colour transfer + point-cloud scene | – | Gate on the evaluator-rendered scene | fail the gate (AA 23.24 < 25.27) → worst |
| 2D colour transfer + vanilla's scene | – | – | −16.9 / −14.1 |
| Transparent / too many Gaussians / NaN scene | – | Scene validation and the gate | all worst |

## Shortcuts left (for the grader), stated plainly

- **`canonical_from_vanilla`: submit another model's Gaussians as `scene`.** The pair is DAVIGS's novel views plus a separately trained vanilla 3DGS scene. It scores **+0.91 (narrow) / +0.40 (broad) dR**: AA 27.37 vs 26.27 dB, with the novel view unchanged. More importantly, it could let a method that corrupts its own scene pass the gate. Images and Gaussians cannot prove they come from one reconstruction, and an image-level transform per pixel can map any render to any target, so no output-only consistency check closes this. The rule is stated in `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md`. A grader can detect it from the method code, which trains a second, independent scene.
- **Hand-writing the generating process.** A method could try to inverse-render the procedural courtyard (11–15 textured primitives, sun, 3 lamps, per-capture ISP) from the visible captures. The generator code and seeds are hidden in `eval/test/`. This was not tested; it would itself be a full inverse-rendering pipeline.

## Headroom check

- No scored metric is at its bound. Reference: NV PSNR 22.87 (cap 100), SSIM 0.73 (bound 1), LPIPS 0.16 (bound 0); AA PSNR 26.27, SSIM 0.74, LPIPS 0.22.
- Two honest variants score above the reference by more than seed noise (`long`, `module_cosine`). The appearance-free scene has room: uv+depth and vanilla render cleaner scenes, so a method that keeps DAVIGS's novel-view quality with a cleaner scene will rank higher.
- There is no brute-force enumeration in this problem: per-scene optimisation from captures.
- The simple baselines score clearly below: vanilla −3.71 / −3.24, 2D copies fail the gate.
- No new experiment was added; the existing two have headroom.

## Not fixed / honest limitations

- **The appearance-free scene of the reproduced DAVIGS is less clean than vanilla 3DGS's and the uv+depth ablation's** (AA LPIPS 0.222 vs 0.147 / 0.124). So the paper's floater/colour-distortion claim and the Table 4 "3D features beat uv+depth" result are not reproduced on this protocol at this scale.
  - The scale is 6k steps instead of 30k, 96×128 images and a 6,000-Gaussian cap.
  - The 1.5× steps variant improves the scene by +1.31 dB / −0.022 LPIPS, so more of the paper's training budget moves it the right way. A full rerun at a larger budget did not fit the compute left.
  - With weight 0.15 the uv+depth ablation ties the reference within seed noise: +0.20 narrow / −0.21 broad, against a seed spread of 0.90 / 0.73. It does not clearly rank below it.
- **The 1 dB gate is tight for variants that trade scene quality for novel-view quality:** `module_cosine` seed 1 passed by 0.07 dB.
- The `long` variant was run for one seed only. `module_cosine` and the reference were run for three.
- Efficiency (training time, VRAM, FPS), PhotoTourism and plug-and-play (Tables 2–3) are not scored.
- Data are synthetic (procedural, ray-traced). A better method here is a better method for global plus 3D-consistent local appearance variation with known cameras, not a validated result on real GLAV captures.
- The job's first allocation was replaced after about 5.4 h, and the relaunch resumed from per-scene caches. Reference outputs were produced once; the frozen predictions and scores were recomputed from them deterministically.
