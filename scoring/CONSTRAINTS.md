# Scientist Constraints

## Data

Use only the inputs in `data/glav_syn/`: training captures, cameras, the left halves of held-out captures and the sparse point cloud. Do not add external images, extra views or labels. Development-scene targets (`eval/train/answers.pt`) may be used for development and tuning on the development scenes only.

## Held-out captures

A held-out capture's left half may only be used to infer appearance (for example per-capture appearance parameters or other appearance-related parameters). It must not supervise scene geometry or scene colour, because that would turn evaluation captures into extra training views.

## Outputs

Write `outputs/<method>/predictions.pt` exactly as described in `briefing/evaluation.md`. The `average_appearance` entry must be the scene rendered without any per-capture appearance state. Do not substitute adapted images, the visible left halves or another model's renders.

## Evaluation

Scores come only from `scripts/evaluate*.sh`, which recompute everything from `predictions.pt`. Writing into `scoring/*.json` has no effect. Comparison methods (`vanilla_3dgs`, `davigs_uv_depth`, `davigs_no_id`) are evaluator-owned and cannot be replaced.

## Scale

Keep the image resolution, cameras and scene set. Changing the optimisation budget is allowed. It is not scored, but report it, since the paper's efficiency claim concerns training cost.

## Compliance Checklist

- [ ] Uses only `data/glav_syn/` inputs
- [ ] Held-out left halves used for appearance only
- [ ] `average_appearance` is a render without per-capture appearance state
- [ ] Scores produced by `scripts/evaluate*.sh` from `predictions.pt`
