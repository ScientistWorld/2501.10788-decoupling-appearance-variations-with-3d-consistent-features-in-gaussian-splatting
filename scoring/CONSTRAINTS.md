# Scientist Constraints

## Data (fixed)

- Use only the benchmark inputs in `data/glav_syn/` (format version 3, `manifest.json`): nine scenes, `dev_00..dev_02` for development and `test_00..test_05` for the hidden test, at 96x128.
- Per scene the inputs are:
  - 30 training captures with their cameras;
  - 6 held-out cameras with the left halves of their captures;
  - 6 canonical-only cameras;
  - the sparse SfM-like point cloud with colours.
- Do not add external images, extra views, labels, depth or pretrained models; none is used by the reference.
- Development-scene targets (`eval/train/answers.pt`) may be used for development and tuning on the development scenes only.
- Nothing under `eval/test/` (generator, seeds, answers, frozen predictions) may be read or used.

## Representation and budget (fixed, as in the reference)

- The scene is a set of 3D Gaussians in the standard 3DGS parameterisation (means, scales, rotations, opacity, degree-3 SH colour), optimised per scene on the provided captures.
- At most 6,000 Gaussians per scene. The evaluator rejects more than 60,000 as a hard guard; 6,000 is the rule.
- At most 6,000 optimisation iterations per scene, each rendering one training capture, as in the reference. An improvement must come from methodology, not from more iterations or Gaussians.
- Test-time inference of a held-out capture's appearance from its left half is allowed in addition. It may update only per-capture appearance parameters.

## Held-out captures

A held-out capture's left half may only be used to infer that capture's appearance. It must not supervise scene geometry, scene colour or any shared appearance component, because that would turn evaluation captures into extra training views.

## Outputs

Write `outputs/<method>/predictions.pt` exactly as described in `briefing/evaluation.md`.

- The `scene` entry must be the same reconstruction that produced the `novel_view` images, as plain 3D Gaussians; the evaluator renders it without any appearance model.
- Do not substitute another model's scene (for example a separately trained plain reconstruction).
- Do not bake held-out left halves into the scene.

## Evaluation

- Scores come only from `scripts/evaluate*.sh`, which recompute everything from `predictions.pt`. Writing into `scoring/*.json` has no effect.
- The comparison methods (`vanilla_3dgs`, `davigs_uv_depth`, `davigs_no_id`) and the gate reference are evaluator-owned and cannot be replaced.

## Compliance Checklist

- [ ] Uses only `data/glav_syn/` inputs (nothing from `eval/test/`, no external data or pretrained models)
- [ ] At most 6,000 Gaussians and 6,000 training iterations per scene
- [ ] Held-out left halves used only for per-capture appearance parameters
- [ ] `scene` is the Gaussian reconstruction behind `novel_view`, not another model's
- [ ] Scores produced by `scripts/evaluate*.sh` from `predictions.pt`
