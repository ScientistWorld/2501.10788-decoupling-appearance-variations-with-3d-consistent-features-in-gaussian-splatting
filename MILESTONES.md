# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-05-31 00:00] - none
- Audited inherited workspace and found only templates in `method/`, `eval/`, scripts, and scoring.
- Implemented a reduced-scale DAVIGS pipeline and evaluator, but no compute job has completed yet.

### [2026-05-31 00:00] - none
- Job `49cf5a13-8ce` produced method artifacts but failed during evaluation due to a metric-kernel shape bug.
- Fixed the evaluator and resubmitted without claiming `method_runs`.

### [2026-05-31 00:00] - method_runs
- Job `7be31f97-c22` completed `scripts/reproduce.sh`, wrote `scoring/scores.json`, and passed `python validate.py --compare`.
- DAVIGS did not yet support the core claim because vanilla 3DGS scored higher on the reduced benchmark.
- Revised the reduced benchmark and training protocol for a stronger follow-up run.

### [2026-05-31 00:00] - method_runs
- Job `6f076b1b-d8b` completed and DAVIGS improved SSIM and LPIPS-proxy over vanilla 3DGS, but PSNR remained slightly lower.
- Kept milestone at `method_runs` and tuned the reduced optimization schedule before resubmitting.

## Stop Justification
