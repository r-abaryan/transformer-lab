# Synthetic safety CI

The [workflow](../../.github/workflows/safety-checks.yml) runs on pull requests, pushes
to `main`, and manual dispatch after the workflow is present on the default branch.
Its matrix covers hosted native Windows and Ubuntu with Python 3.12 and 3.14. It uses
only the standard library: no pip installs, training, datasets, checkpoints or archived
experiment discovery. Runner labels and Python minor versions receive upstream updates;
this is a compatibility matrix, not an exact reproduction of the research environment.

Run the same driver locally from the repository root:

```text
python tests/ci_safety.py
```

The driver runs targeted runner tests, targeted CI-gate tests, the named native-junction
regression, and the `runner-safety` CLI. It checks command exit codes and the CLI JSON
summary: source hashes/inventory, integrity checks, nonzero test count, zero training,
no failures/exceptions, and consistent skip details. Windows may not skip the actual
junction test. Known symlink-privilege skips remain visible and are allowed; they are
not reported as passing coverage. Linux must explicitly report the Windows-only skip.
The named junction invocation also fails if that regression is missing.

Each matrix job uploads its generated `results/test_runs/ci_*` logs and reports even
after a test failure, with seven-day retention. Artifacts include `ci-report.json`,
`runner-summary.json` when readable, command logs, and the raw CLI run summary. A job
that cannot create any reports fails artifact upload; no persistence guarantee is made
for abrupt termination or unavailable storage. Do not restore untrusted artifacts as code.

The workflow uses `pull_request`, not `pull_request_target`, read-only repository
permissions, no persisted checkout credentials, pinned action commits, a ten-minute job
timeout and cancellation of superseded runs. Initial fork runs or workflow changes may
require repository approval, and the GitHub connector may require additional workflow
write permission. Repository branch-protection requirements are not changed by this PR.

Green synthetic CI is not a filesystem sandbox, archived-copy verification, historical
reproduction, or scientific confirmation. Existing disposable-checkout guidance remains
in the [hardened-runner guide](safe-test-runner.md). Native-Windows validation is only
established by inspecting an actual successful Windows run and its skip report, not by
adding this workflow or passing the summary-gate unit tests on Linux.
