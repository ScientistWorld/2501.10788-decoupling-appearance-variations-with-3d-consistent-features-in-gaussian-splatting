# Audit plan and continuation

Base commit: `5f64f875bd408f276c471b79f230ed6de52b942b`.

The resumed job runs 28 cases in one CPU job. All generation and inference
occur there. Review `results_table.txt`, `audit_summary.json`, and per-case
`run_record.json` after completion. All prediction/score mutations are restored
in `finally`; no evaluator source is changed. A failed evaluator has its old
test-score file removed first, except the explicitly tested direct-score-write
case, which verifies that prewritten scores get overwritten.

Honest cases: three reference optimization seeds on fixed scene seed 7; double
training/adaptation; cosine learning-rate decay at the original budget for three
seeds; 50-step weak model; no-appearance weak model. Original baseline/ablation
outputs remain fixed in these comparisons.

Shortcut cases cover visible generator answers, exhaustive seed search, target
replacement, constraint failure with perfect counterfeit targets, baseline
replacement, missing and negative metadata, failed-method omission through a
visible score plan, an unlisted method on disk, shrinking/reordering the split,
direct score writing, NaNs, empty/missing artifacts, constants and infinities.

After the job, diagnose any failed harness/quality checks and iterate as needed.
Write final `REDTEAM.md` with quality/rank first, seed spread and reproduction
check second, shortcut table third. Numeric ranks are cohort dependent; always
retain raw metric values and distinguish reference test scores from paper values.
Restore `scripts/run.sh` with git from the base commit and restore any accidentally
changed non-audit files. Do not touch pre-existing system-owned README/manifest/
trajectory changes. Commit only intended audit files and the script restoration.
Set action.yaml to done only when the report is finished.

First job `df73767e-40f` was cancelled after repeated worker-drain preemptions
restarted the bundle. The saved reference and seed 7/8/9 all completed. Fixed
scene seed spread: PSNR 20.4150--23.8100 (range 3.39497 dB), much larger than
the saved reference's 1.22583 dB improvement over the baseline. The resumed job
skips completed cases, runs shortcuts first, saves atomic training/embedding
optimizer checkpoints every 100 updates, and caches generated honest outputs.
The additional honest local-SSIM variant repairs the global-SSIM training loss.
