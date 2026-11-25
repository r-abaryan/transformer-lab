# Fresh-clone quickstart

This guide covers the lightweight runner checks and copy/continuation validation path.
It is not a reproduction guide for every historical study. Read [AGENTS.md](../../AGENTS.md)
before changing research code, protocols or evidence.

## 1. Clone with Git

Use Git with repository access and an available Python 3.12 interpreter. Python 3.12 is
the shared version exercised by the hosted Windows/Ubuntu checks for the paths below;
it is not the recorded environment of every historical experiment.

```text
git clone https://github.com/r-abaryan/transformer-mechanism.git
cd transformer-mechanism
python --version
```

Use your selected interpreter consistently if its command is `python3` or another name.
An activated virtual environment is fine; no packages need installing for the next two
checks. A local `.venv-gpu` directory is not part of the clone and is not required here.

Use a Git checkout, not a source ZIP, for byte-hash validation. The scoped
[Git attributes](../../.gitattributes) materialize the three continuation input files
with their historical CRLF serialization. Fresh cloning is a convenient way to apply
that checkout policy; it does not depend on merging an unrelated CI PR. Existing
working copies may retain earlier line endings. See the [serialization guide](continuation-artifact-validation.md)
before investigating a hash mismatch; do not replace recorded checksums or force-refresh
files containing local work just to obtain a pass.

## 2. Check the runner's safeguards

From the repository root:

```text
python tests/safe_runner.py --suite runner-safety
```

This standard-library suite needs no GPU, PyTorch, dataset or restored checkpoint. It
uses synthetic files to check path handling, failure reporting and integrity safeguards;
it does not verify Transformer equations or replay research. Platform/privilege skips
are reported explicitly and are not passing coverage.

A successful invocation exits zero and writes a summary in a fresh ignored folder under
`results/test_runs/`. Those retained run summaries are separate from disposable synthetic
test fixtures. The [hardened-runner guide](safe-test-runner.md) defines the protection
scope and limitations. Hashing detects persistent changes; it is not a filesystem sandbox.

## 3. Validate the committed continuation artifacts

```text
python experiments/validate_copy_exposure_continuation.py
```

This standard-library command reads the committed `plan.json`, `report.json` and
`metrics.csv` under `results/copy_exposure_continuation/`. It needs no binary checkpoints,
PyTorch or training. It writes no artifact files and prints a JSON result and input hashes.
A successful check reports `passed: true`, `runs: 8` and `csv_rows: 328` and exits zero.
A stored `repair: false` or `exposure_specific_benefit: false` is a scientific outcome,
not automatically a validation failure. Do not redirect output onto historical evidence.

This checks artifact structure and scalar consistency, not saved-model predictions,
gradients, training execution or architecture superiority. See the
[validator guide](continuation-artifact-validation.md) for the exact schema, gain
reconciliation, tolerance boundaries, Git checkout regression command and hosted evidence.

## Reproduction matrix

| Operation | Inputs/environment | What it establishes |
|---|---|---|
| `runner-safety` above | Git checkout and Python standard library; synthetic fixtures only | Runner safeguards within the tested scope, not model correctness |
| Continuation validator above | Three committed text artifacts, their checkout serialization and Python standard library | Structural/scalar consistency and referenced byte hashes, not model replay |
| `archived-copy` initialized correctness checks | PyTorch, selected archived modules, and frozen source/protocol/result directories inspected by the launcher | Tested equations, gradients and evaluator behavior on fresh initialized fixtures; no training or full-project coverage |
| Saved-checkpoint diagnostics/audits | Study-specific dependencies plus the exact saved weights, tensors and datasets required by that audit | Only the predictions/arithmetic and provenance explicitly checked by that audit |
| Formal training reproduction | Frozen protocol, matching environment/data, seeds, declared budget, required start state and appropriate hardware | A separately executed study; not implied by any quickstart check |

Only the first two rows are the minimal fresh-clone path here. Git-based checkout-policy
regressions additionally need Git; other suites have their own prerequisites and some
use short synthetic optimizer runs. Do not generalize the zero-training scope of these
two commands to every file under `tests/`.

## Beyond the lightweight checks

For the selected archived copy checks, first read the [test instructions](../../tests/README.md)
and [runner guide](safe-test-runner.md), prepare the recorded dependency environment,
and inspect required paths. In a disposable checkout with those prerequisites:

```text
python tests/safe_runner.py --suite archived-copy
```

Use that explicit suite, not blanket discovery or direct execution of the historical
test scripts. Some historical entry points write fixed preflight/checkpoint paths.
The older `tests/run_tests.py` remains preserved; the hardened entry point is preferred
for the selected suites. No claim is made that every historical test works from Git alone.

[CPU](../../requirements-cpu.txt), [GPU](../../requirements-gpu.txt),
[data](../../requirements-data.txt) and [analysis](../../requirements-analysis.txt)
requirements record separate environments/dependencies. [pyproject.toml](../../pyproject.toml)
exposes those same pinned sets as mutually exclusive `cpu` / `gpu` extras (plus `analysis`
and `data`) for `uv sync --extra ...`; avoid bare `uv sync` without `--extra` against an
already-populated `.venv`, as it removes unselected extras. In particular, the CPU file
records an observed Windows CPython environment, not a universal installation guarantee
for every platform. Do not install the GPU stack merely to run the lightweight checks.

The [.gitignore](../../.gitignore) excludes binary `.pt` checkpoints/tensors, dataset
copies, logs and selected large dumps. The `results/` directory is therefore not a
complete trained-model archive. For a saved-model audit, inspect its protocol and
provenance records first, identify missing files, and restore the exact artifacts from
your retained archive. Do not invent substitute files or silently rerun training to
replace missing evidence. This guide adds no download service or new artifact manifest.

Before launching training, follow the [research workflow](../research-plan.md) and the
study's frozen protocol. Preserve executed sources/results and respect fixed-path
recording behavior. A successful text-artifact check is not authorization for a new run.
