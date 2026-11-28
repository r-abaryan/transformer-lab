"""Formal C/D execution runner (decoder FFN frozen) and pre-execution verification."""
import argparse
import copy
import csv
import hashlib
import importlib
import json
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("experiments", "tests"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import audit_copy_decoder_ffn_adaptation as audit_mod
from copy_decoder_ffn_adaptation import (
    FFN_NAMES,
    check_frozen,
    check_order,
    full_forward_check,
    process_memory,
    rejects,
    require,
    restore,
    same,
    snapshot,
    targets,
)
from copy_position_sites import digest, evaluate, forward, inputs, make_model, measures, positions, sample, write
from prepare_copy_decoder_ffn_adaptation import (
    CONTINUATION_REL,
    DATA_REL,
    DEFAULT_LOCK_NAME,
    STAGE_EXAMPLES,
    STAGE_TARGETS,
    STAGE_UPDATES,
    STAGES,
    START_REL,
    _norm_rel,
    require_debug_assertions,
    verify_historical_lineage,
    verify_lock_bundle,
    verify_protected_hashes,
)
from safe_runner import fresh_directory, validate_target
from validate_copy_exposure_continuation import COUNTS, METRICS, decode_json

RESERVED_OUTPUT_NAMES = frozenset({
    "copy_decoder_ffn_adaptation",
    "copy_exposure_continuation",
    "copy_learning_trajectory",
    "copy_position_sites",
    "test_runs",
})
CD_CELLS = (("C", "cross_only"), ("D", "both"))


def _audit_dir_relative(out_dir, root):
    return out_dir.relative_to(root).as_posix()


def _check_windows_memory(mem_before, mem_after):
    if sys.platform == "win32":
        require(mem_before.get("available") and mem_after.get("available")
                and mem_after.get("peak_working_set_bytes", 0) > 0,
                "Windows process memory measurement failed")


def _paired_summary(rows, first_outputs, second_outputs):
    require(len(rows) == len(first_outputs) == len(second_outputs), "Paired sequence count mismatch")
    both_pass = rescued = harmed = both_fail = first_ok = second_ok = 0
    for row, seq_a, seq_b in zip(rows, first_outputs, second_outputs):
        target = row + [2]
        hit_a = seq_a == target
        hit_b = seq_b == target
        first_ok += int(hit_a)
        second_ok += int(hit_b)
        if hit_a and hit_b:
            both_pass += 1
        elif (not hit_a) and hit_b:
            rescued += 1
        elif hit_a and (not hit_b):
            harmed += 1
        else:
            both_fail += 1
    total = len(rows)
    return dict(
        sequences=total,
        first_correct=first_ok,
        second_correct=second_ok,
        both_pass=both_pass,
        rescued=rescued,
        harmed=harmed,
        both_fail=both_fail,
        net_gain=(second_ok - first_ok) / total if total else 0.0,
    )


def compute_quartet_decision(root, ab_report, cd_report, gates_spec, stages, expected_counts, eval_pools):
    """Evaluate all locked step-final quartet decision rules and short-preservation conditions."""
    final_step = stages[-1]
    runs_final = {
        "A": next(r for r in ab_report["runs"] if r["arm"] == "cross_only" and r["step"] == final_step),
        "B": next(r for r in ab_report["runs"] if r["arm"] == "both" and r["step"] == final_step),
        "C": next(r for r in cd_report["runs"] if r["cell"] == "C" and r["step"] == final_step),
        "D": next(r for r in cd_report["runs"] if r["cell"] == "D" and r["step"] == final_step),
    }
    acc10 = {c: r["evaluation"]["metrics"]["test_10"]["all"]["exact_accuracy"] for c, r in runs_final.items()}
    ab_diff = acc10["B"] - acc10["A"]
    dc_diff = acc10["D"] - acc10["C"]
    interaction = ab_diff - dc_diff

    gen10 = {
        c: torch.load(root / _norm_rel(r["evaluation"]["scores"]), weights_only=True)["test_10"]["generated"]
        for c, r in runs_final.items()
    }
    paired10 = {
        f"{x}_vs_{y}": _paired_summary(eval_pools["test_10"], gen10[x], gen10[y])
        for x, y in (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"))
    }

    short_pools = [p for p in expected_counts if int(p.rsplit("_", 1)[1]) <= 6]
    if "short_preservation_pools" in gates_spec:
        require(set(short_pools) == set(gates_spec["short_preservation_pools"]),
                "Short preservation pools mismatch")
    ab_lookup = {(r["arm"], r["step"]): r["evaluation"]["metrics"] for r in ab_report["runs"]}
    cd_lookup = {(r["cell"], r["step"]): r["evaluation"]["metrics"] for r in cd_report["runs"]}
    short_failures = []
    short_passed_by_cell = {c: True for c in ("A", "B", "C", "D")}
    worst_short = 1.0
    for step in [6000, *stages]:
        per_cell = (
            {
                "A": ab_report["initial"]["metrics"],
                "B": ab_report["initial"]["metrics"],
                "C": cd_report["initial"]["metrics"],
                "D": cd_report["initial"]["metrics"],
            }
            if step == 6000
            else {"A": ab_lookup[("cross_only", step)], "B": ab_lookup[("both", step)],
                  "C": cd_lookup[("C", step)], "D": cd_lookup[("D", step)]}
        )
        for cell_name, metrics in per_cell.items():
            for pool_name in short_pools:
                for grp in ("distinct", "repeated"):
                    val = metrics[pool_name][grp]["exact_accuracy"]
                    worst_short = min(worst_short, val)
                    if val < gates_spec["short_preservation_min_accuracy"]:
                        short_passed_by_cell[cell_name] = False
                        short_failures.append(dict(cell=cell_name, step=step, pool=pool_name, group=grp, exact_accuracy=val))

    final_short_ok = {
        c: all(
            runs_final[c]["evaluation"]["metrics"][pool_name][grp]["exact_accuracy"]
            >= gates_spec["short_preservation_min_accuracy"]
            for pool_name in short_pools
            for grp in ("distinct", "repeated")
        )
        for c in ("A", "B", "C", "D")
    }

    gen8 = {
        c: torch.load(root / _norm_rel(r["evaluation"]["scores"]), weights_only=True)["test_8"]["generated"]
        for c, r in runs_final.items()
    }
    rows8 = eval_pools["test_8"]
    len8 = {}
    for grp in ("distinct", "repeated", "all"):
        ids = [i for i, row in enumerate(rows8) if grp == "all" or (len(set(row)) == len(row)) == (grp == "distinct")]
        sub_rows = [rows8[i] for i in ids]
        g_acc = {c: r["evaluation"]["metrics"]["test_8"][grp]["exact_accuracy"] for c, r in runs_final.items()}
        len8[grp] = dict(
            accuracy=g_acc,
            b_minus_a=g_acc["B"] - g_acc["A"],
            d_minus_c=g_acc["D"] - g_acc["C"],
            interaction=(g_acc["B"] - g_acc["A"]) - (g_acc["D"] - g_acc["C"]),
            paired={
                f"{x}_vs_{y}": _paired_summary(sub_rows, [gen8[x][i] for i in ids], [gen8[y][i] for i in ids])
                for x, y in (("A", "B"), ("C", "D"))
            },
        )

    repair = {
        c: final_short_ok[c] and all(
            runs_final[c]["evaluation"]["metrics"][pool][grp][metric] >= gates_spec["historical_repair_threshold"]
            for pool, grp in (("test_8", "distinct"), ("test_8", "repeated"), ("test_10", "repeated"))
            for metric in ("exact_accuracy", "on_time_eos")
        )
        for c in ("A", "B", "C", "D")
    }
    contrast_ok = interaction >= gates_spec["material_attenuation_min_gain"] and ab_diff > 0.0
    return dict(
        decision_step=final_step,
        primary=dict(
            accuracy=acc10,
            b_minus_a=ab_diff,
            d_minus_c=dc_diff,
            interaction=interaction,
            material_attenuation_min_gain=gates_spec["material_attenuation_min_gain"],
            contrast_threshold_met=contrast_ok,
            material_attenuation=contrast_ok and (not short_failures),
            positive_exposure_benefit=ab_diff > 0.0,
            paired=paired10,
        ),
        short_preservation=dict(
            min_accuracy=gates_spec["short_preservation_min_accuracy"],
            worst_accuracy=worst_short,
            passed=not short_failures,
            passed_by_cell=short_passed_by_cell,
            final_step_passed_by_cell=final_short_ok,
            failures=short_failures,
        ),
        secondary_length8=len8,
        historical_repair_passed=repair,
    )


def evaluate_and_audit_checkpoint(model, pools_map, dest_scores_path, metrics_auditor):
    """Evaluate all non-uniform pools under contiguous cross-only wiring and run independent audit."""
    require(not dest_scores_path.exists(), f"Refusing to overwrite scores: {dest_scores_path}")
    ms = {}
    saved = {}
    max_err = 0.0
    model.eval()
    for pool_name, rows in pools_map.items():
        m, s = evaluate(model, rows, "cross_only")
        err, _ = metrics_auditor.audit_supplied_cell(model, rows, "cross_only", s, m)
        max_err = max(max_err, err)
        ms[pool_name] = m
        saved[pool_name] = s
    torch.save(saved, dest_scores_path)
    return dict(
        metrics=ms,
        scores=dest_scores_path,
        scores_sha256=digest(dest_scores_path),
        max_reference_error=max_err,
    )


def train_frozen_segment(model, opt, gen, pg, train_pools, begin_step, stop_step, arm, frozen_snapshot):
    """Run one frozen-FFN continuation segment with independent RNG and per-step freeze verification."""
    require(arm in ("cross_only", "both"), f"Unknown arm: {arm}")
    require(stop_step > begin_step, "stop_step must exceed begin_step")
    check_order(model, opt)
    ffn_params = targets(model)
    require(all(not p.requires_grad and p.grad is None for p in ffn_params.values()),
            "Decoder FFN parameters must be frozen before training")

    rg = torch.Generator()
    rg.set_state(gen.get_state().clone())
    rpg = torch.Generator()
    rpg.set_state(pg.get_state().clone())

    stream = hashlib.sha256()
    pstream = hashlib.sha256()
    examples = 0
    targets_count = 0
    model.train()
    for step in range(begin_step + 1, stop_step + 1):
        n, idx, packed = sample(train_pools, gen, "mixed", step)
        pos = positions(n, pg)

        expected_n = 3 + (step - 1) % 4
        expected_batch = 840 // (expected_n + 1)
        half = len(train_pools[f"train_{expected_n}"]) // 2
        ranks = torch.rand(2, 105, generator=rg, dtype=torch.float64)
        expected_idx = torch.floor(ranks[:, : expected_batch // 2] * half).long()
        expected_idx[1] += half
        expected_idx = expected_idx.flatten()
        expected_pos = torch.randperm(11, generator=rpg)[: expected_n + 1].sort().values

        require(n == expected_n and torch.equal(idx, expected_idx)
                and packed == struct.pack("210d", *ranks.flatten().tolist())
                and torch.equal(pos, expected_pos), f"Sampler mismatch at step {step}")
        require(torch.equal(gen.get_state(), rg.get_state())
                and torch.equal(pg.get_state(), rpg.get_state()), f"RNG state mismatch at step {step}")

        stream.update(json.dumps([n, idx.tolist()]).encode())
        pstream.update(json.dumps(pos.tolist()).encode())

        x = train_pools[f"train_{n}"][idx]
        prefix, y = inputs(x)
        opt.zero_grad(set_to_none=True)
        z = forward(model, x, prefix, pos if arm == "both" else None, pos)[0]
        loss = F.cross_entropy(z.flatten(0, 1), y.flatten())
        require(bool(torch.isfinite(loss)), f"Nonfinite loss at step {step}")
        loss.backward()

        for name, param in model.named_parameters():
            if name in FFN_NAMES:
                require(param.grad is None, f"Frozen FFN parameter received gradient at step {step}: {name}")
            else:
                require(param.grad is not None and bool(torch.isfinite(param.grad).all()),
                        f"Missing or nonfinite trainable gradient at step {step}: {name}")

        opt.step()
        check_frozen(model, opt, frozen_snapshot)
        examples += len(x)
        targets_count += y.numel()

    checkpoint = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(opt.state_dict()),
        step=stop_step,
        generator=gen.get_state().clone(),
        position_generator=pg.get_state().clone(),
    )
    return stream.hexdigest(), pstream.hexdigest(), examples, targets_count, checkpoint


def runner_preflight(root=ROOT, lock_name=DEFAULT_LOCK_NAME, check_lock=True, save_dir=None):
    """Exercise evaluator, independent audit entry, supplied logits, reload path, and freeze error controls."""
    require_debug_assertions()
    audit_mod.validate_lock_name(lock_name)
    tick = time.perf_counter()
    mem_before = process_memory()
    root = Path(root).resolve(strict=True)
    reloaded_auditor = importlib.reload(audit_mod)
    if check_lock:
        lock_status = verify_lock_bundle(root=root, lock_name=lock_name)
    else:
        lock_status = dict(passed=True, skipped=True)

    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix="copy-ffn-runner-pre-") as tmp:
        work = Path(tmp)
        torch.manual_seed(2027)
        factory = lambda: make_model("rope").eval()
        init_model = factory()
        init_path = work / "initialized.pt"
        torch.save(init_model.state_dict(), init_path)

        reloaded_model = make_model("rope").eval()
        reloaded_model.load_state_dict(torch.load(init_path, weights_only=True), strict=True)
        rows = [[3, 4, 5], [3, 3, 4]]
        metrics, saved = evaluate(reloaded_model, rows, "cross_only")
        scores_path = work / "init_scores.pt"
        torch.save(saved, scores_path)
        loaded_scores = torch.load(scores_path, weights_only=True)
        err, merr = reloaded_auditor.audit_supplied_cell(reloaded_model, rows, "cross_only", loaded_scores, metrics)

        # Missing group and wrong generated/logit/scores negative controls on independent audit entry.
        missing_group_metrics = copy.deepcopy(metrics)
        missing_group_metrics.pop("distinct")
        try:
            reloaded_auditor.audit_supplied_cell(reloaded_model, rows, "cross_only", loaded_scores, missing_group_metrics)
        except (AssertionError, KeyError):
            pass
        else:
            raise AssertionError("Independent auditor accepted missing group")

        bad_gen = copy.deepcopy(loaded_scores)
        bad_gen["generated"][0] = [2] if bad_gen["generated"][0] != [2] else [3, 2]
        rejects(lambda: reloaded_auditor.audit_supplied_cell(reloaded_model, rows, "cross_only", bad_gen, metrics))

        bad_logits = copy.deepcopy(loaded_scores)
        bad_logits["logits"] = bad_logits["logits"].clone()
        bad_logits["logits"].add_(1.0)
        rejects(lambda: reloaded_auditor.audit_supplied_cell(reloaded_model, rows, "cross_only", bad_logits, metrics))

        bad_scores = copy.deepcopy(loaded_scores)
        bad_scores["scores"] = bad_scores["scores"].clone()
        bad_scores["scores"].add_(1.0)
        rejects(lambda: reloaded_auditor.audit_supplied_cell(reloaded_model, rows, "cross_only", bad_scores, metrics))

        # Supplied logits scoring check for correct vs wrong generations.
        labels = torch.tensor([r + [2] for r in rows])
        logits = torch.full((2, 4, 16), -8.0, dtype=torch.float64)
        logits.scatter_(-1, labels[..., None], 8.0)
        scores = torch.zeros((2, 2, 4), dtype=torch.float64)
        for is_correct in (True, False):
            gen_seqs = [r + [2] for r in rows] if is_correct else [[2], [2]]
            sup_metrics = measures(rows, logits, scores, gen_seqs)
            reloaded_auditor._strict_close(reloaded_auditor.own_measures(rows, logits, scores, gen_seqs), sup_metrics, "supplied")
            require(sup_metrics["all"]["exact_accuracy"] == int(is_correct), "Supplied logit scoring mismatch")

        # Self-exposure wiring must diverge from cross-only in forward pass and trained updates.
        x_t = torch.tensor(rows)
        prefix, labels_t = inputs(x_t)
        pos_t = torch.tensor([0, 3, 7, 10])
        with torch.no_grad():
            z_cross = forward(reloaded_model, x_t, prefix, None, pos_t)[0]
            z_self = forward(reloaded_model, x_t, prefix, pos_t, pos_t)[0]
            require(float((z_self - z_cross).abs().max()) > 1e-5, "Self-exposure forward logits failed to diverge")

        # Freeze, Adam moment/step tampering, and gradient detachment negative controls.
        warm_model = factory()
        warm_opt = torch.optim.Adam(warm_model.parameters(), lr=0.001)
        warm_opt.zero_grad(set_to_none=True)
        F.cross_entropy(forward(warm_model, x_t, prefix, None, pos_t)[0].flatten(0, 1), labels_t.flatten()).backward()
        warm_opt.step()
        warm_cp = dict(
            model=copy.deepcopy(warm_model.state_dict()),
            optimizer=copy.deepcopy(warm_opt.state_dict()),
            step=1,
            generator=torch.Generator().manual_seed(11).get_state(),
            position_generator=torch.Generator().manual_seed(12).get_state(),
        )
        grad_checks = [full_forward_check(warm_cp, factory, x_t, pos_t, exp) for exp in (False, True)]

        frozen_pair = {}
        for exp in (False, True):
            fm, fo, _ = restore(warm_cp, factory, frozen=True)
            fsnap = snapshot(fm, fo)
            fo.zero_grad(set_to_none=True)
            F.cross_entropy(forward(fm, x_t, prefix, pos_t if exp else None, pos_t)[0].flatten(0, 1),
                            labels_t.flatten()).backward()
            fo.step()
            check_frozen(fm, fo, fsnap)
            frozen_pair[exp] = (fm, fo)
        require(any(not torch.equal(v, frozen_pair[True][0].state_dict()[k])
                    for k, v in frozen_pair[False][0].state_dict().items()),
                "Self-exposure failed to diverge frozen-FFN step updates")

        # Mutate frozen weight and Adam state inside independent audit check -> must fail closed.
        frozen_m, frozen_o = frozen_pair[True]
        mutated_cp = dict(
            model=copy.deepcopy(frozen_m.state_dict()),
            optimizer=copy.deepcopy(frozen_o.state_dict()),
            step=2,
            generator=warm_cp["generator"].clone(),
            position_generator=warm_cp["position_generator"].clone(),
        )
        start_snap, _ = reloaded_auditor.capture_ffn_snapshot(warm_cp)
        reloaded_auditor.verify_frozen_checkpoint(mutated_cp, warm_cp, start_snap, expected_step=2, expected_start_step=1)

        bad_weight_cp = copy.deepcopy(mutated_cp)
        with torch.no_grad():
            bad_weight_cp["model"][FFN_NAMES[0]].add_(1e-9)
        rejects(lambda: reloaded_auditor.verify_frozen_checkpoint(bad_weight_cp, warm_cp, start_snap, 2, 1))

        bad_adam_cp = copy.deepcopy(mutated_cp)
        ffn_pid = start_snap[FFN_NAMES[0]]["param_id"]
        bad_adam_cp["optimizer"]["state"][ffn_pid]["step"] = bad_adam_cp["optimizer"]["state"][ffn_pid]["step"] + 1
        rejects(lambda: reloaded_auditor.verify_frozen_checkpoint(bad_adam_cp, warm_cp, start_snap, 2, 1))

        # Authorization refusal negative control.
        try:
            run_formal("unauthorized_probe", authorize=False, root=root, lock_name=lock_name)
        except PermissionError:
            pass
        else:
            raise AssertionError("Unauthorized formal run was accepted")

    mem_after = process_memory()
    _check_windows_memory(mem_before, mem_after)
    report = dict(
        passed=True,
        formal_ready=False,
        formal_authorized=False,
        formal_updates=0,
        lock_verified=lock_status.get("passed", False),
        auditor_reloaded=True,
        max_reference_error=max(err, merr),
        checks=dict(
            initialized_saved_checkpoint_audited=True,
            supplied_logits_scored=True,
            missing_group_rejected=True,
            corrupted_generation_logits_scores_rejected=True,
            self_exposure_divergence_verified=True,
            frozen_ffn_weight_and_adam_corruption_rejected=True,
            unauthorized_run_rejected=True,
        ),
        gradient_checks=grad_checks,
        memory=dict(before=mem_before, after=mem_after),
        seconds=time.perf_counter() - tick,
        environment=audit_mod.collect_runtime_environment(),
        limitation="Pre-execution runner/auditor contract check only; zero formal training updates.",
    )
    if save_dir is not None:
        write(Path(save_dir) / "runner_preflight.json", report)
    return report


@dataclass(frozen=True)
class _RunnerOverrides:
    verify_runtime: bool = True
    verify_history: bool = True
    check_ab_checkpoints: bool = True
    expected_counts: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(COUNTS))
    expected_stages: tuple[int, ...] = STAGES
    expected_stage_updates: int = STAGE_UPDATES
    expected_stage_examples: int = STAGE_EXAMPLES
    expected_stage_targets: int = STAGE_TARGETS


def _run_formal_impl(
    output_name,
    authorize=False,
    root=ROOT,
    lock_name=DEFAULT_LOCK_NAME,
    overrides=None,
):
    """Validate lock, configuration, history, and preflight first, then run authorized C/D and independent audit."""
    require_debug_assertions()
    audit_mod.validate_lock_name(lock_name)
    if authorize is not True:
        raise PermissionError("Formal C/D execution requires explicit authorize=True (--authorize-formal-execution).")
    require(isinstance(output_name, str), "Output directory name must be string")
    if (output_name.casefold() in RESERVED_OUTPUT_NAMES
            or output_name.casefold() in audit_mod.WINDOWS_RESERVED_NAMES
            or output_name.casefold() == lock_name.casefold()):
        raise ValueError(f"Reserved output directory cannot be used for formal runs: {output_name}")

    root = Path(root).resolve(strict=True)
    results_root = root / "results"
    validate_target(results_root, output_name, root=root)

    tick = time.perf_counter()
    mem_before = process_memory()

    lock_summary = verify_lock_bundle(root=root, lock_name=lock_name)
    lock_path = root / "results" / lock_name / "lock.json"
    lock = decode_json(lock_path.read_text(encoding="utf-8"))
    require(list(FFN_NAMES) == lock["budget"]["frozen_parameters"], "FFN_NAMES mismatch with locked budget")

    if overrides is None:
        sched = audit_mod.derive_locked_schedule(lock, COUNTS)
        verify_runtime = True
        verify_history = True
        check_ab_checkpoints = True
        expected_counts = sched["counts"]
        expected_stages = sched["stages"]
        expected_stage_updates = sched["stage_updates"]
        expected_stage_examples = sched["stage_examples"]
        expected_stage_targets = sched["stage_targets"]
    else:
        verify_runtime = overrides.verify_runtime
        verify_history = overrides.verify_history
        check_ab_checkpoints = overrides.check_ab_checkpoints
        expected_counts = overrides.expected_counts
        expected_stages = tuple(overrides.expected_stages)
        expected_stage_updates = overrides.expected_stage_updates
        expected_stage_examples = overrides.expected_stage_examples
        expected_stage_targets = overrides.expected_stage_targets
        audit_mod._validate_fixture_schedule(
            expected_counts, expected_stages, expected_stage_updates,
            expected_stage_examples, expected_stage_targets,
        )

    if verify_runtime:
        expected_torch = lock["environment"]["torch"]
        require(torch.__version__ == expected_torch,
                f"Runtime torch {torch.__version__} != locked historical A/B replay torch {expected_torch}")
        require(torch.get_num_threads() == lock["budget"]["threads"]
                and torch.are_deterministic_algorithms_enabled() is lock["budget"]["deterministic_algorithms"],
                "Formal run requires locked CPU thread count and deterministic algorithms setting")

    if verify_history:
        lineage = verify_historical_lineage(root)
        protected_hashes = lineage["protected_hashes"]
    else:
        protected_hashes = lock["protected_hashes"]
        verify_protected_hashes(root, protected_hashes)
    verify_protected_hashes(root, lock["study_hashes"])
    verify_protected_hashes(root, lock["historical_artifact_hashes"])

    ab_report_path = root / _norm_rel(f"{CONTINUATION_REL}/report.json")
    require(digest(ab_report_path) == lock["historical_artifact_hashes"][_norm_rel(f"{CONTINUATION_REL}/report.json")],
            "Historical A/B report SHA-256 mismatch")
    ab_report = json.loads(ab_report_path.read_text(encoding="utf-8"))
    audit_mod.validate_ab_report_contract(
        root,
        ab_report,
        lock["gates"],
        expected_counts,
        expected_stages,
        expected_stage_updates,
        expected_stage_examples,
        expected_stage_targets,
        expected_hist_hashes=lock["historical_artifact_hashes"] if check_ab_checkpoints else None,
    )

    data_raw = json.loads((root / DATA_REL).read_text(encoding="utf-8"))
    eval_pools = {k: v for k, v in data_raw["pools"].items() if not k.startswith("uniform_")}
    require(digest(root / DATA_REL) == lock["historical_artifact_hashes"][DATA_REL], "Data SHA-256 mismatch")
    audit_mod.validate_data_pools(eval_pools, expected_counts=expected_counts)
    train_pools = {k: torch.tensor(v) for k, v in eval_pools.items() if k.startswith("train_")}

    start_path = root / START_REL
    start_sha = digest(start_path)
    require(start_sha == lock["historical_artifact_hashes"][START_REL], "Start checkpoint SHA-256 mismatch")
    start_cp = torch.load(start_path, weights_only=True)
    require(type(start_cp.get("step")) is int and start_cp.get("step") == 6000, "Start checkpoint step must be 6000")
    audit_mod.capture_ffn_snapshot(start_cp)

    execution_code_hashes = {rel: digest(root / rel) for rel in audit_mod.EXECUTION_CODE_FILES}
    pre_report = runner_preflight(root=root, lock_name=lock_name, check_lock=True)

    # Create exclusive output directory only after all pre-execution checks succeed.
    out_dir = fresh_directory(results_root, output_name, root=root)
    write(out_dir / "preflight.json", pre_report)

    plan = dict(
        lock_sha256=lock_summary["lock_sha256"],
        start=6000,
        stages=list(expected_stages),
        cells=dict(C=lock["budget"]["cells"]["C"], D=lock["budget"]["cells"]["D"]),
        frozen_parameters=list(FFN_NAMES),
        protected_hashes=protected_hashes,
        execution_code_hashes=execution_code_hashes,
        historical_ab_report_sha256=digest(ab_report_path),
        preflight_sha256=digest(out_dir / "preflight.json"),
    )
    write(out_dir / "plan.json", plan)

    start_model, _, _ = restore(start_cp, lambda: make_model("rope"), frozen=True)
    initial_ev = evaluate_and_audit_checkpoint(start_model, eval_pools, out_dir / "3767_frozen_6000_scores.pt", audit_mod)
    initial_record = dict(
        metrics=initial_ev["metrics"],
        scores=_audit_dir_relative(initial_ev["scores"], root),
        scores_sha256=initial_ev["scores_sha256"],
        max_reference_error=initial_ev["max_reference_error"],
    )

    hist_init = torch.load(root / _norm_rel(ab_report["initial"]["scores"]), weights_only=True)
    init_saved = torch.load(initial_ev["scores"], weights_only=True)
    for pool_name in eval_pools:
        require(torch.equal(init_saved[pool_name]["logits"], hist_init[pool_name]["logits"])
                and torch.equal(init_saved[pool_name]["scores"], hist_init[pool_name]["scores"])
                and init_saved[pool_name]["generated"] == hist_init[pool_name]["generated"],
                f"Initial evaluation mismatch vs historical A/B at {pool_name}")

    report = dict(
        status="running",
        plan_sha256=digest(out_dir / "plan.json"),
        initial=initial_record,
        runs=[],
    )
    write(out_dir / "report.json", report)

    cell_state = {}
    start_ffn_snapshots = {}
    prior_shas = {}
    for cell, arm in CD_CELLS:
        model, opt, (gen, pg) = restore(start_cp, lambda: make_model("rope"), frozen=True)
        cell_state[cell] = dict(arm=arm, model=model, opt=opt, gen=gen, pg=pg)
        start_ffn_snapshots[cell] = snapshot(model, opt)
        prior_shas[cell] = start_sha

    prev_step = 6000
    for stop in expected_stages:
        stage_pair = []
        ab_stage = {
            arm: next(r for r in ab_report["runs"] if r["arm"] == arm and r["step"] == stop)
            for arm in ("cross_only", "both")
        }
        for cell, arm in CD_CELLS:
            st = cell_state[cell]
            stream_hex, pstream_hex, examples, targets_count, cp = train_frozen_segment(
                st["model"], st["opt"], st["gen"], st["pg"],
                train_pools, prev_step, stop, arm, start_ffn_snapshots[cell],
            )
            seg_updates = stop - prev_step
            require(seg_updates == expected_stage_updates == ab_stage[arm]["updates"]
                    and examples == expected_stage_examples == ab_stage[arm]["examples"]
                    and targets_count == expected_stage_targets == ab_stage[arm]["active_targets"],
                    f"Segment accounting mismatch at {cell}@{stop}")
            require(stream_hex == ab_stage[arm]["stream"] and pstream_hex == ab_stage[arm]["position_stream"],
                    f"Paired stream mismatch against historical A/B at {cell}@{stop}")
            if check_ab_checkpoints:
                ab_ckpt_rel = _norm_rel(ab_stage[arm]["checkpoint"])
                ab_ckpt_path = root / ab_ckpt_rel
                require(digest(ab_ckpt_path) == ab_stage[arm]["checkpoint_sha256"],
                        f"Historical A/B checkpoint SHA-256 mismatch: {ab_ckpt_rel}")
                ab_cp = torch.load(ab_ckpt_path, weights_only=True)
                require(torch.equal(cp["generator"], ab_cp["generator"])
                        and torch.equal(cp["position_generator"], ab_cp["position_generator"]),
                        f"Saved RNG state diverged from historical A/B checkpoint at {cell}@{stop}")

            ckpt_path = out_dir / f"3767_{cell}_{arm}_{stop}.pt"
            require(not ckpt_path.exists(), f"Refusing to overwrite checkpoint: {ckpt_path}")
            torch.save(cp, ckpt_path)
            eval_out = evaluate_and_audit_checkpoint(
                st["model"], eval_pools, out_dir / f"3767_{cell}_{arm}_{stop}_scores.pt", audit_mod
            )
            run_record = dict(
                cell=cell,
                arm=arm,
                decoder_ffn_frozen=True,
                step=stop,
                start_sha256=prior_shas[cell],
                checkpoint=_audit_dir_relative(ckpt_path, root),
                checkpoint_sha256=digest(ckpt_path),
                stream=stream_hex,
                position_stream=pstream_hex,
                updates=seg_updates,
                examples=examples,
                active_targets=targets_count,
                evaluation=dict(
                    metrics=eval_out["metrics"],
                    scores=_audit_dir_relative(eval_out["scores"], root),
                    scores_sha256=eval_out["scores_sha256"],
                    max_reference_error=eval_out["max_reference_error"],
                ),
            )
            report["runs"].append(run_record)
            prior_shas[cell] = run_record["checkpoint_sha256"]
            stage_pair.append((cp, run_record))
            write(out_dir / "report.json", report)

        require(torch.equal(stage_pair[0][0]["generator"], stage_pair[1][0]["generator"])
                and torch.equal(stage_pair[0][0]["position_generator"], stage_pair[1][0]["position_generator"]),
                f"C/D RNG mismatch at step {stop}")
        require(not same(stage_pair[0][0]["model"], stage_pair[1][0]["model"]),
                f"C and D models failed to diverge at step {stop}")
        prev_step = stop

    flat_rows = [
        dict(cell=r["cell"], arm=r["arm"], step=r["step"], pool=pool_name, group=group_name, **metrics_dict)
        for r in report["runs"]
        for pool_name, groups in r["evaluation"]["metrics"].items()
        for group_name, metrics_dict in groups.items()
        if metrics_dict is not None
    ]
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell", "arm", "step", "pool", "group", *METRICS])
        writer.writeheader()
        writer.writerows(flat_rows)

    verify_protected_hashes(root, protected_hashes)
    verify_protected_hashes(root, execution_code_hashes)
    verify_protected_hashes(root, lock["study_hashes"])
    verify_protected_hashes(root, lock["historical_artifact_hashes"])

    decision = compute_quartet_decision(
        root, ab_report, report, lock["gates"], expected_stages, expected_counts, eval_pools
    )
    for cell, arm in (("A", "cross_only"), ("B", "both")):
        require(decision["historical_repair_passed"][cell] == ab_report["gates"][arm]["repair"],
                f"Quartet historical_repair_passed[{cell}] diverged from historical A/B report gates")

    mem_after = process_memory()
    _check_windows_memory(mem_before, mem_after)
    report.update(
        status="completed_unaudited",
        acceptance_policy="Formal acceptance requires audit.json with passed=true, formal_acceptance=true, and report_sha256 matching this file.",
        formal_updates=sum(r["updates"] for r in report["runs"]),
        decision=decision,
        metrics_sha256=digest(csv_path),
        memory=dict(before=mem_before, after=mem_after),
        seconds=time.perf_counter() - tick,
        protected_files=len(protected_hashes),
        environment=audit_mod.collect_runtime_environment(),
    )
    write(out_dir / "report.json", report)

    try:
        if overrides is None:
            audit_summary = audit_mod.audit_directory(
                out_dir, root=root, lock_name=lock_name, save=True, diagnostic_cross_runtime=False
            )
            acceptance = audit_mod.verify_acceptance(
                out_dir, root=root, lock_name=lock_name, reaudit=False
            )
        else:
            audit_summary = audit_mod._audit_directory_for_tests(
                out_dir,
                root=root,
                lock_name=lock_name,
                ab_report_rel=f"{CONTINUATION_REL}/report.json",
                expected_counts=expected_counts,
                expected_stages=expected_stages,
                expected_stage_updates=expected_stage_updates,
                expected_stage_examples=expected_stage_examples,
                expected_stage_targets=expected_stage_targets,
                verify_lock=True,
                check_ab_checkpoints=check_ab_checkpoints,
                save=True,
                diagnostic_cross_runtime=False,
            )
            acceptance = audit_mod._verify_acceptance_for_tests(
                out_dir,
                root=root,
                lock_name=lock_name,
                ab_report_rel=f"{CONTINUATION_REL}/report.json",
                expected_counts=expected_counts,
                expected_stages=expected_stages,
                expected_stage_updates=expected_stage_updates,
                expected_stage_examples=expected_stage_examples,
                expected_stage_targets=expected_stage_targets,
                verify_lock=True,
                check_ab_checkpoints=check_ab_checkpoints,
                reaudit=False,
            )
    except Exception as exc:
        report["status"] = "audit_failed"
        report["audit_error"] = f"{type(exc).__name__}: {exc}"
        write(out_dir / "report.json", report)
        raise

    return dict(
        passed=True,
        accepted=acceptance["accepted"],
        output_directory=_audit_dir_relative(out_dir, root),
        formal_updates=report["formal_updates"],
        report_sha256=acceptance["report_sha256"],
        audit_sha256=acceptance["audit_sha256"],
        decision=decision,
        audit=audit_summary,
    )


def run_formal(output_name, authorize=False, root=ROOT, lock_name=DEFAULT_LOCK_NAME):
    """Execute authorized formal C/D strictly under verified locked schedule and environment."""
    return _run_formal_impl(
        output_name,
        authorize=authorize,
        root=root,
        lock_name=lock_name,
        overrides=None,
    )


def _run_formal_for_tests(
    output_name,
    authorize=False,
    root=ROOT,
    lock_name=DEFAULT_LOCK_NAME,
    verify_runtime=True,
    verify_history=True,
    check_ab_checkpoints=True,
    expected_counts=COUNTS,
    expected_stages=STAGES,
    expected_stage_updates=STAGE_UPDATES,
    expected_stage_examples=STAGE_EXAMPLES,
    expected_stage_targets=STAGE_TARGETS,
):
    return _run_formal_impl(
        output_name,
        authorize=authorize,
        root=root,
        lock_name=lock_name,
        overrides=_RunnerOverrides(
            verify_runtime=verify_runtime,
            verify_history=verify_history,
            check_ab_checkpoints=check_ab_checkpoints,
            expected_counts=dict(expected_counts),
            expected_stages=tuple(expected_stages),
            expected_stage_updates=expected_stage_updates,
            expected_stage_examples=expected_stage_examples,
            expected_stage_targets=expected_stage_targets,
        ),
    )


def main(argv=None):
    require_debug_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "verify-ready", "run"))
    parser.add_argument("--lock-name", default=DEFAULT_LOCK_NAME)
    parser.add_argument("--save-run-name", default=None,
                        help="Optional fresh folder under results/test_runs/ for preflight report")
    parser.add_argument("--output-name", default=None,
                        help="Required exclusive folder under results/ for formal C/D run")
    parser.add_argument("--authorize-formal-execution", action="store_true",
                        help="Explicitly authorize formal C/D training updates")
    args = parser.parse_args(argv)

    if args.action != "preflight" and args.save_run_name is not None:
        parser.error("--save-run-name is only supported with preflight")
    if args.action != "run" and (args.output_name is not None or args.authorize_formal_execution):
        parser.error("--output-name and --authorize-formal-execution are only supported with run")

    audit_mod.validate_lock_name(args.lock_name)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if args.action == "preflight":
        if args.save_run_name is not None:
            validate_target(ROOT / "results/test_runs", args.save_run_name, root=ROOT)
        result = runner_preflight(ROOT, lock_name=args.lock_name, check_lock=True, save_dir=None)
        if args.save_run_name is not None:
            save_dir = fresh_directory(ROOT / "results/test_runs", args.save_run_name, root=ROOT)
            write(save_dir / "runner_preflight.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    if args.action == "verify-ready":
        lock_summary = verify_lock_bundle(root=ROOT, lock_name=args.lock_name)
        lock = decode_json((ROOT / "results" / args.lock_name / "lock.json").read_text(encoding="utf-8"))
        runtime_ok = torch.__version__ == lock["environment"]["torch"]
        lineage = verify_historical_lineage(ROOT)
        verify_protected_hashes(ROOT, lock["historical_artifact_hashes"])
        verify_protected_hashes(ROOT, lock["protected_hashes"])
        pre = runner_preflight(ROOT, lock_name=args.lock_name, check_lock=True)
        print(json.dumps(dict(
            passed=pre["passed"] and lineage["passed"] and runtime_ok,
            ready_for_authorization=pre["passed"] and lineage["passed"] and runtime_ok,
            formal_ready=False,
            formal_authorized=False,
            formal_updates=0,
            runtime_matches_locked_torch=runtime_ok,
            actual_torch=torch.__version__,
            locked_torch=lock["environment"]["torch"],
            lock=lock_summary,
            lineage_protected_files=lineage["protected_count"],
            preflight=pre,
        ), indent=2, allow_nan=False))
        return 0 if runtime_ok else 1

    if not args.output_name:
        parser.error("run requires --output-name")
    if not args.authorize_formal_execution:
        parser.error("run requires --authorize-formal-execution")
    result = run_formal(
        args.output_name,
        authorize=args.authorize_formal_execution,
        root=ROOT,
        lock_name=args.lock_name,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
