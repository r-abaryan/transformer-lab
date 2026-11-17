# Decoder FFN adaptation and positional exposure

Status: proposed protocol and initialized-fixture preflight, 2026-11-17.
No formal runner, frozen execution plan, study results or training authorization.
Base reviewed: `dc84d618c2ddc6c1ee6b6406fe43c97a14107627`.

## Question and evidence boundary

Does allowing the decoder FFN parameters to update modify the full-copy benefit
of adding self-site positional exposure from the same weak checkpoint?
The [continuation](../topics/benchmarks/copy-exposure-continuation-results.md)
records 55/128 versus 99/128 complete length-10 copies; the 95% repair gate failed.
The [trace and classifier control](../topics/benchmarks/copy-exposure-stage-trace-results.md)
show downstream growth in a readout projection, not FFN adaptation causality.
Read the [failure register](../assessments/failure-register.md): incompatible
component swaps are not a clean substitute for this training comparison.

## Proposed factorial comparison

| Cell | Self-site exposure | Decoder FFN updates |
|---|---|---|
| A | No, retain cross-only | Allowed |
| B | Yes, retain cross exposure too | Allowed |
| C | No, retain cross-only | Frozen |
| D | Yes, retain cross exposure too | Frozen |

Use seed 3767 cross-only update 6000 from `results/copy_learning_trajectory`.
Restore identical model tensors, complete Adam state, example-generator state and
position-generator state. No new initialization, optimizer reset or seed search.
Freeze exactly `base.decoder.linear{1,2}.{weight,bias}` in C/D. All attention,
encoder FFN, normalization and classifier parameters stay trainable. Keep the
same ordered optimizer parameter collection when loading its state. Frozen
parameters must have `grad is None`; parameters, moments and step counters must
remain bitwise unchanged. Gradients through FFN inputs must remain available.

Preserve the historical content/EOS objective and coefficients. In B/D the same
sampled ordered positions apply jointly to encoder and decoder self-attention;
all cells use the same cross-attention positions. This does not localize the two
self-attention sites or separate upstream, normalization and readout adaptation.

## Fixed candidate budget and measurements

Proposed window: updates 6001-8000, Adam lr .001, CPU float64, one thread,
no dropout or accumulation. Cycle lengths 3/4/5/6 with batches 210/168/140/120:
840 supervised targets per update, 319000 examples and 1680000 targets per cell;
8000 updates, 1276000 examples and 6720000 targets across the quartet.
Record every checkpoint at 6000/6500/7000/7500/8000. Only 8000 decides outcomes;
no best-checkpoint selection, early outcome selection, extension or replacement.

Reuse exactly the 14 non-uniform pools in `copy_position_sites/data.json`.
Freeze the data/start/dependency hashes before any formal evaluation. Record
train/validation/test lengths 3-6 and tests 8/10, all/distinct/repeated groups,
with absent groups explicitly null. Evaluation uses contiguous positions and
self-generated sequences, the original EOS stop rule and length+3 cap. Record
complete copies, content exactness, teacher-forced CE and token/content accuracy,
and early/on-time/late/missing EOS. Correct-prefix scores are not generation.

Primary contrast is complete length-10 copy accuracy, (B-A)-(D-C).
Report raw counts, paired rescues/harms and both exposure effects as well.
Proposed material attenuation threshold: at least 13/128 (10.15625 percentage
points) in this interaction, with B-A positive and every short group in all
four cells >=99% complete copies at every measured checkpoint. This is a declared
practical threshold, not a power calculation or statistical-significance claim.
Report length-8 distinct/repeated and aggregate effects as secondary checks;
do not replace a failing primary result with a favorable secondary metric.

An attenuated effect with preservation supports a conditional contribution of
allowing FFN adaptation, not unique mediation or FFN-only sufficiency. A positive
D-C demonstrates benefit is possible with the starting decoder FFN fixed; it
does not prove equivalence or absence of FFN involvement in the original route.
Short-task damage makes specific mediation interpretation inconclusive. Report
negative interactions and mixed outcomes. Keep the historical >=95% long-group
repair gate separate and unchanged. Counts are not independent training seeds.
Opened data and one selected start make this a diagnostic, not fresh confirmation.

## Preflight and execution gates

Run only these explicit new entry points; no blanket historical discovery:

```sh
python -m unittest discover -s tests -p test_copy_decoder_ffn_adaptation.py -v
python experiments/copy_decoder_ffn_adaptation.py preflight
```

The module has only `preflight`, never `run`. It creates initialized fixtures in
a temporary directory, reloads them with weights_only, exercises the native
evaluator and reference audit, correct/wrong supplied generations, missing-group
and corruption controls, native/reference trainable gradients, frozen Adam history,
input gradients and matched fixture streams. It makes 17 discarded fixture
updates, not 8000 formal updates. Historical checkpoints/data are not opened.
Profiles cover the four cells and four full batch sizes, including Adam state;
CPU peak memory is explicitly unmeasured, not inferred from tensor-byte counts.
The reference shares historical model helpers; it is not an independent model.

Before any formal run, a separate reviewed implementation/approval must:
- Pass this preflight and its tests on native Windows, profile peak process memory
  including optimizer/evaluation/import overhead, and record environment versions.
- Verify availability and historical hashes of the trained start, pools, old
  plan/report/audit and dependencies. Do not regenerate missing evidence.
- Independently verify historical sampling and generator restoration, all labels,
  counters, stream hashes, reload/import paths and intentional failure controls.
- Reproduce A/B's declared historical predictions and endpoint model/optimizer/
  generator tensors exactly under the declared environment; stop on mismatch.
- Freeze the final hashes, budget, thresholds and preservation gates prospectively.
- Audit all saved cells, gradients/state-freeze invariants and complete metrics;
  guard historical evidence and write only to a new, exclusive study directory.

Any nonfinite value, failed invariant or historical-replay mismatch stops as an
implementation/precondition failure, without tuning or replacement. A short-fit
failure is reported, not repaired by another budget. No automatic seed, block,
site or range sweep. Fresh confirmation would require a separate decision.

## Primary sources and interpretation

[Ruoss et al. (2023), section 3](https://arxiv.org/html/2305.16843v1#S3)
was inspected for ordered positional sampling. Exposure is published; this
checkpoint, RoPE-site choice, factorial contrast and thresholds are local choices.
[Heimersheim and Nanda (2024)](https://arxiv.org/html/2404.15255v1)
was inspected for intervention/metric interpretation limits. This is constrained
training, not activation patching. Neither source predicts the local counts,
establishes this hypothesis or supplies a novelty claim. No field-level novelty
or reliable complete repair is established by this proposed diagnostic.
