# Decoder FFN adaptation: proposed protocol and preflight

2026-11-17. No formal runner, execution authorization or new research result.
Baseline hosted Windows/Python 3.12 passed at `667562c`; PR #5 merged at `61a6f73`.
Review CI evidence, exact diffs and raw outputs before authorizing any formal run.

## Scientific design (unchanged)

Does allowing decoder FFN updates change the incremental complete-copy benefit
of added self-site positional exposure from the same selected weak checkpoint?

| Cell | Added self-site exposure | Decoder FFN updates |
|---|---|---|
| A | No | Allowed |
| B | Yes | Allowed |
| C | No | Frozen |
| D | Yes | Frozen |

Use the seed-3767 cross-only update-6000 model, complete Adam state and both RNGs
from `results/copy_learning_trajectory`. All cells retain cross-site exposure;
B/D also apply the same positions jointly at both self-attention sites.
Freeze only `base.decoder.linear{1,2}.{weight,bias}`; keep every other parameter
trainable, original single Adam group/order and moment mapping intact. Frozen
gradients must be None; tensors, moments and counters must be bitwise unchanged.

Inherit labels, pools, objective, evaluation/stopping and complete metrics from
[the continuation protocol](copy-exposure-continuation.md): 14 non-uniform pools;
all short train/validation/test groups, long 8/10 groups, absent groups null;
full generation, content, teacher-forced CE/accuracy and all EOS timing outcomes.
Use native contiguous-position evaluation, not oracle prefixes or forced EOS.
Window 6001-8000; Adam lr .001; CPU float64, one thread, no dropout/accumulation.
Cycle lengths 3/4/5/6 with batches 210/168/140/120, 840 content/EOS targets/update.
Per cell: 2000 updates, 319000 examples, 1680000 targets. Quartet maximum: 8000
updates, 1276000 examples, 6720000 targets. Observe 6000/6500/7000/7500/8000;
only 8000 decides outcomes. No tuning, favorable checkpoint, seed or budget sweep.

Primary: (B-A)-(D-C) on all 128 length-10 complete copies. Report both effects,
counts and paired rescues/harms. Material attenuation: >=13/128, positive B-A,
and >=99% short-group complete copying in every cell at every measured checkpoint.
This is a proposed practical threshold, not a power/significance claim. Report
length-8/group effects secondarily and the historical 95% repair gate separately.
Short preservation is an interpretability condition, not implementation validity:
correct freezing that harms familiar copying is a reportable, mechanistically
inconclusive outcome, not permission to discard or repair it. Report small,
negative and mixed results. State/stream/gradient defects are implementation
failures. Attenuation is not unique mediation; positive D-C is not equivalence.
Opened pools, one selected start and four arms are not independent replications.
See [results](../topics/benchmarks/copy-exposure-continuation-results.md),
[trace limits](../topics/benchmarks/copy-exposure-stage-trace-results.md) and
[failure register](../assessments/failure-register.md); do not repeat failed swaps.

## Preflight acceptance

```sh
python -m unittest discover -s tests -p test_copy_decoder_ffn_adaptation.py -v
python experiments/copy_decoder_ffn_adaptation.py preflight
```

All 11 tests and additions must execute without unexpected skips. Each preflight
makes 17 discarded fixture updates, 16 profiles, 10 audited cells; the suite and
CLI each invoke it and unit fixtures add steps. Report accounting separately.
Require passed=true and formal_ready=false. The CLI has only preflight, never run.

Compare identical full-forward frozen/unrestricted values and shared gradients
at absolute 1e-10, rtol 0. Retain the actual FFN output gradient. Deliberate FFN
output detachment must preserve values but change upstream gradients and fail
the same comparison; residual-bypass gradients alone do not suffice. Keep all
native/reference gradient, saved initialized evaluator/audit, supplied generation,
missing-group and corruption checks. Reject live/saved optimizer permutations,
each frozen tensor/moment/counter mutation, and zero-gradient Adam updates.
Byte comparisons distinguish signed zero. The optimizer-ID check assumes the
historical sequential-ID, single-group format, not generic checkpoint support.
Preserve saved fixture bytes, original in-memory state and restored generators.

## Sampler fixture is not historical replay

Use historical sample/positions helpers on synthetic ordered pools. Independently
check indices, packed ranks, positions, RNG states, hashes and midstream restore;
reject an extra RNG draw even if returned indices match. No historical data or
checkpoint is loaded. A later runner must separately preserve and verify:
- Hashed update-6000 model, complete Adam state, both saved RNGs and lineage.
- Exact training pool identities/order: distinct first half, repeated second.
- n=3+(step-1)%4, batch=840//(n+1), fixed 2x105 float64 torch.rand consumption
  every step (including unused columns), floor(rank*half), second-half offset,
  then first-half/second-half flattening of indices.
- Separate randperm(11), first n+1 entries sorted; BOS prefixes, full content/EOS
  labels, CE weights, counters, checkpoint steps, resume and generator states.
- Continuation hashes: default json.dumps([n, idx.tolist()]).encode() and
  json.dumps(pos.tolist()).encode(); whitespace/serialization is significant.

Synthetic success does not certify saved pools/checkpoints or A/B trajectories.
Exact A/B replay remains mandatory before C/D. When running historical verification,
`replay-ab` performs real optimizer steps in memory across 6001-8000 ("read-only"
refers to preserving all historical files untouched on disk, not zero computation
or zero replay updates; formal C/D updates remain 0 until separately authorized).
After separate authorization, A/B are the first 4000 of the 8000 quartet updates,
not extra unbudgeted runs. Stop missing artifacts/mismatches; never regenerate
historical evidence.

## Memory, evidence and later execution

Windows stdlib ctypes GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX records
before/after peak/current working set and peak/current private commit in bytes.
These are distinct process counters, not tensor totals. Windows measurement
failure fails validation; unsupported platforms report unavailable. Peaks cover
process lifetime (imports, optimizer, evaluation), not isolated-step increments
or formal-run upper bounds. Prefer the fresh CLI measurement. Record batch 210,
step/total timings, CPU/platform/Python/PyTorch and numerical settings.

Approved Windows CI saves logs, JSON, environment, commands/timings and source/
evidence hashes under ignored results/test_runs and reviewer-accessible artifacts
(30-day retention; archive before expiry). Hosted Windows is not the user's
laptop. Review raw evidence; shared model helpers are not an independent model.
Before formal execution: review this diff/CI evidence, intended-hardware feasibility,
historical hashes/sampler/labels/counters/RNG restoration and exact A/B replay;
review a saved-result audit and new exclusive contained output directory; freeze
final hashes/metrics/gates/budget; obtain separate approval. No automatic merge,
formal training, architecture expansion or weakened checks belongs in this PR.

## Sources inspected

[Ruoss et al., section 3](https://arxiv.org/html/2305.16843v1#S3): ordered positional
exposure. [Heimersheim and Nanda](https://arxiv.org/html/2404.15255v1): intervention/
metric interpretation. [Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex):
working-set/commit semantics. Checkpoint/site/contrast/threshold choices and
scores are local, not source predictions. No novelty or reliable repair claim.
