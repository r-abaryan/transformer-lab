# Understanding Transformer mechanisms

We study Transformer mathematics and what small trained models actually compute. Tests, controlled interventions and independent audits support bounded claims. They do not establish an overall architecture winner or a novel replacement.

Read the [current research checkpoint](docs/assessments/research-checkpoint.md) for status and remaining questions. The [project quality review](docs/assessments/project-quality-review.md) records what was checked and its limits. Earlier milestones are in [research history](docs/assessments/research-history.md).

The focus is [explaining learned failures](docs/assessments/learning-mechanism-agenda.md). The [copy assessment](docs/assessments/copy-generalization-assessment.md) connects inference-time diagnostics to a learned positional-exposure repair and its limits. The [retention synthesis](docs/assessments/retention-learning-causal-account.md) preserves that branch's established effects and unresolved causes. No experiment is currently queued; the [RL roadmap](docs/topics/rl/rl-roadmap.md) remains parked.

| Entry | Purpose |
|---|---|
| [Documentation by topic](docs/topics/README.md) | Attention, FFNs, normalization, positions, embeddings and benchmarks |
| [Educational guides](docs/guides/README.md) | Short explanations, equations and examples |
| [Study reports](docs/results/README.md) | Methods, measured findings and limitations |
| [Assessments](docs/assessments/README.md) | Conclusions across studies |
| [Sources and claims](docs/reference/README.md) | Primary papers and claim boundaries |
| [Protocols](docs/protocols/README.md) and [decisions](docs/decisions/README.md) | Frozen designs and rationale |
| [Research workflow](docs/research-plan.md) | How questions become tests and studies |
| [Rules](AGENTS.md) and [failure register](docs/assessments/failure-register.md) | Prevent repeat mistakes |

## Quickstart

Start with a Git clone and Python 3.12. From the repository root, run the lightweight
checks below; no GPU, PyTorch, datasets or restored checkpoints are required:

```text
python tests/safe_runner.py --suite runner-safety
python experiments/validate_copy_exposure_continuation.py
```

The first checks synthetic runner safeguards and creates a fresh ignored run summary.
The second is read-only: it checks committed continuation JSON/CSV artifacts and their
hashes, not model predictions or training. Use a Git checkout so the scoped artifact
line-ending policy applies. Neither command establishes a learned mechanism or a new
trained-model result.

Follow the [fresh-clone guide and reproduction matrix](docs/guides/fresh-clone-quickstart.md)
for expected outputs, artifact restoration and the separate prerequisites of archived
copy checks. The [hardened-runner guide](docs/guides/safe-test-runner.md),
[continuation validator guide](docs/guides/continuation-artifact-validation.md) and
[test instructions](tests/README.md) define their scopes. Historical standalone scripts
can write fixed evidence paths; do not run them indiscriminately.

`experiments/` contains executed implementations and independent audits. `results/` contains raw measurements/checkpoints; `docs/` contains explanations and reports. Dependencies are recorded in [requirements-cpu.txt](requirements-cpu.txt), [requirements-gpu.txt](requirements-gpu.txt), [requirements-data.txt](requirements-data.txt) and [requirements-analysis.txt](requirements-analysis.txt). GPU work uses native Windows on an RTX 4060 laptop with 8 GB VRAM.

The project began on 2026-09-29. Frozen evidence remains at its original paths; document cleanup does not change the measurements.
