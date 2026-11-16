# Supplemental continuation artifact validation

This standard-library-only validator checks the structural and scalar consistency of
`results/copy_exposure_continuation/{plan.json,report.json,metrics.csv}`. It does not
replace or modify the [frozen auditor](../../experiments/audit_copy_exposure_continuation.py),
producer, protocol, checkpoints, audit records or reported results.

From the repository root:

```text
python experiments/validate_copy_exposure_continuation.py
python -m unittest discover -s tests -p test_continuation_artifact_validation.py -v
```

Use `--directory PATH` to inspect a separate artifact fixture. The validator reads only
the three named files and prints JSON to stdout, or a failure object to stderr with
nonzero exit. It creates no result directories and writes no artifact files. Avoid
redirecting shell output onto historical evidence. No PyTorch or checkpoint is loaded.

## Contract and checks

The schema is deliberately specific to the frozen 2026-10-08 continuation: two arms,
four saved stages, fourteen pools, and the inspected cohort counts. Pool/group/metric
names and counts come from the executed producer and stored report/CSV layout, not a
new experimental protocol. The [original protocol](../protocols/copy-exposure-continuation.md)
remains authoritative for the study design. Each run has 41 non-null metric rows;
eight runs require 328 unique `(arm, step, pool, group)` records. Initial metrics are
validated separately and do not belong in the CSV. Each pool must include all three
group keys; `test_10/distinct` is explicitly null, not an omitted record or zero score.

Checks reject duplicate/missing/unexpected runs and records, duplicate CSV headers or
JSON keys, ragged rows, incorrect metric fields/counts, nonfinite/out-of-range values,
and inconsistent EOS partitions. CSV metrics are compared to the report with absolute
tolerance 1e-10; counts are exact integers. Repair thresholds remain >=99% short and
>=95% long, with the stored >=10-point benefit criterion evaluated without tolerance.
Plan/CSV byte hashes must match their report references. Other protected-file manifests,
checkpoint hashes, predictions, gradients and training histories are not replayed.

## Gain reconciliation

The producer computes length-10 gain from `repeated`; the original auditor uses `all`.
For this frozen cohort, `distinct` is null and both populated groups contain all 128
sequences. The supplement checks identical group counts/metrics, calculates both gains,
and checks each against the recorded gain. It does not silently substitute one cohort
or change a failed repair decision into a successful one. This equivalence is specific
to this dataset and must not be generalized to future mixed cohorts.

## Frozen checkout serialization

The historical plan checksum expects Windows CRLF bytes. The first Ubuntu job read
LF bytes and correctly stopped at `Plan hash mismatch`; its structural and boundary
tests passed. Reconstructing the complete plan and verifying its Git blob reproduced
both hashes exactly: LF `1cf044abdd0394609386198d65e29881bcc27468510a025e9178a5e050d5724e`,
CRLF `ef8030b315090fed1688bd314850401bc11fbea1dbea4c998af103525855f3e1`.

The root `.gitattributes` now declares `text eol=crlf` only for this continuation's
`plan.json`, `report.json` and `metrics.csv`. Fresh Git checkout materializes their
original serialization on either platform; their stored Git blobs and recorded
checksums are not rewritten. No blanket rule, renormalization, checksum replacement or
fallback normalization inside the validator is introduced. Existing materialized files
may not be rewritten just because attributes change: prefer a fresh disposable clone,
and do not delete or force-refresh an existing working copy without protecting local work.

Run the isolated Git regression tests with:

```text
python -m unittest discover -s tests -p test_continuation_checkout_policy.py -v
```

These require Git, otherwise explicitly skip. They check the three exact path rules,
fresh CRLF materialization with core.autocrlf false/true, unchanged Git blobs, a clean
artifact diff, and rejection of missing-policy/content-change checksum controls. They
use temporary repositories only and do not refresh any real project artifact.

## Boundary tests and hosted verification

```text
python -m unittest discover -s tests -p test_continuation_validation_boundaries.py -v
```

The added boundary suite directly tests the unrounded benefit predicate at 0.10 and
its adjacent representable floats, and checks that failed repair blocks benefit.
End-to-end fixtures separately check gains computed by subtracting two accuracies;
for example, `1.0 - 0.9` is below the float literal `0.10` and is not rounded up.
Loss and accuracy cross-checks exercise both perturbation signs inside and outside
1e-10, nearest serialized values straddling that boundary, and the inclusive comparator
at exactly 1e-10. Boundary fixtures reuse the schema; original structural tests remain
separate and unchanged.

The [dedicated workflow](../../.github/workflows/continuation-validation.yml) runs the
original structural tests, boundary and checkout-policy tests, and the validator over the complete
committed artifacts on hosted Ubuntu and native Windows with Python 3.12. It requires
successful commands, 8 runs, 328 rows, zero training and matching input hashes. It also
compares before/after hashes of the three artifact files, original producer, auditor,
protocol and audit.json. Per artifact, it records Git blob identity/digest, checkout
digest, CRLF counts and Git attributes; it requires exact historical hash matches.
Plan/CSV references come from report.json; the report reference comes from audit.json. These are integrity checks, not a filesystem sandbox or a complete audit
of every historical dependency. Failed scientific repair gates remain valid results;
validation success does not require a successful learned repair.

Evidence is written outside the checkout under runner temporary storage, displayed in
the Actions job summary, and retained as an artifact for seven days even after normal
test failures. It includes command logs, input/protected-file hashes, and the validator
result when successful. Abrupt termination or unavailable storage can prevent evidence
persistence. The job has read-only repository permissions, no stored checkout credentials,
pinned actions and no training/dependency installs. It does not post comments or merge PRs.
PR #2's separate safety workflow is unchanged.

## Validation limits

The initial 13 structural tests and six new boundary methods have been exercised on
synthetic fixtures during development, not full historical checkpoints. Large connector
responses truncated the historical report/CSV, so full-artifact success and Windows
compatibility must be established from actual successful hosted runs and their hashes.
The earlier Ubuntu mismatch is diagnosed; complete success after the checkout-policy
correction still requires both new hosted jobs to pass. Adding the policy alone does not establish it. Inspect that evidence before merging.
Any mismatch must be investigated without rewriting frozen evidence to force a pass.
No checkpoint, prediction, gradient or training replay is performed.
