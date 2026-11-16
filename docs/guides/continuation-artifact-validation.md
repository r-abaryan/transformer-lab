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

## Validation limits

Synthetic fixtures exercise the read-only CLI and corruption rejection, including a
same-length duplicate-row replacement that passes the old per-row-plus-count logic.
They are not trained-model evidence or a full rerun of the historical audit. Preparation
inspected the producer, protocol and available report/CSV content, but large connector
responses were truncated; no successful run over the full committed artifacts is claimed.
Run the first command against a complete clone to obtain that separate validation.
Native Windows remains untested for this supplement. PR #2's safety CI is independent;
this PR does not modify its workflow or claim this new suite is already included there.
