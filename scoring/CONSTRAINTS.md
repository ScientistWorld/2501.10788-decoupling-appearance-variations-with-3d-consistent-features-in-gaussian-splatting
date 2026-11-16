# Scientist Constraints

## Data

Use only the inputs in `data/glav_syn/`: training captures, cameras, the left halves of held-out captures, the canonical cameras and the sparse point cloud. Do not add external images, extra views or labels. Development-scene targets (`eval/train/answers.pt`) may be used for development and tuning on the development scenes only. Nothing under `eval/test/` may be read or used.

## Held-out captures

A held-out capture's left half may only be used to infer that capture's appearance (for example per-capture appearance parameters). It must not supervise scene geometry or scene colour, because that would turn evaluation captures into extra training views.

## Outputs

Write `outputs/<method>/predictions.pt` exactly as described in `briefing/evaluation.md`. The `scene` entry must be the same reconstruction that produced the `novel_view` images, as plain 3D Gaussians; the evaluator renders it without any appearance model. Do not substitute another model's scene (for example a separately trained plain reconstruction), and do not bake held-out left halves into it.

## Evaluation

Scores come only from `scripts/evaluate*.sh`, which recompute everything from `predictions.pt`. Writing into `scoring/*.json` has no effect. Comparison methods (`vanilla_3dgs`, `davigs_uv_depth`, `davigs_no_id`) and the gate reference are evaluator-owned and cannot be replaced.

## Scale

Keep the image resolution, cameras and scene set. Changing the optimisation budget is allowed. It is not scored, but report it, since the paper's efficiency claim concerns training cost. The evaluator accepts at most 60,000 Gaussians per scene.

## Compliance Checklist

- [ ] Uses only `data/glav_syn/` inputs (nothing from `eval/test/`)
- [ ] Held-out left halves used for appearance only
- [ ] `scene` is the Gaussian reconstruction behind `novel_view`, not another model's
- [ ] Scores produced by `scripts/evaluate*.sh` from `predictions.pt`
