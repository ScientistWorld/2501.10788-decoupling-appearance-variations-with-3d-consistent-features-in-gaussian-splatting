# Scoring audit

Every case has a `run.sh` that installs its method/artifact, invokes the visible
train scorer, and then invokes **exactly** `bash scripts/evaluate_test.sh` with no
arguments in a separate process. The evaluator and Python installation are never
changed. Original outputs and score files are restored after each case.

Run the whole CPU audit with `bash redteam/run_all.sh` in the existing container.
No packages, network, GPUs, or background processes are needed. Method generation
is compute work and must run in a submitted job. The fixed-scene optimization
seeds are 7, 8, 9; the dataset seed is always 7.

The saved reference fixture is reusable as authorized in the audit request. It is
stored in ignored `.state/`. For a fresh clone, first reproduce the normal outputs
using the original reproduction script. No shortcut implementation reads the
paper, hidden code, reference metric specifications, or hidden scores. Audit
analysis uses those files only to assess the results after evaluation.

Each case keeps `scores_test.json`, `scores_train.json`, and `run_record.json`.
Rejected cases record nonzero exit codes; their absence of scores is never
mistaken for a successful run. Logs and prediction tensors are ignored.

`analyze.py` records an illustrative rank using exactly the directions, absolute
coefficient weights, experiment weights, and missing-value skip rule from the
request. Its cohort normalization is explicitly documented; unknown future
attempts can change numerical ranks. Raw metric vectors and pool-independent
Pareto comparisons are the primary evidence.
