"""Independent saved-checkpoint, bitwise FFN-freeze, stream, and quartet-gate auditor."""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("experiments", "tests"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from audit_copy_position_exposure import own_measures
from audit_copy_position_sites import audit_cell
from copy_decoder_ffn_adaptation import FFN_NAMES, check_order, require, same
from copy_position_sites import digest, make_model, write
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
    verify_lock_bundle,
    verify_protected_hashes,
)
from safe_runner import checked_directory, redirected
from validate_copy_exposure_continuation import COUNTS, GROUPS, METRICS, decode_json

PAIR_LABELS = (
    ("both_pass", (True, True)),
    ("rescued", (False, True)),
    ("harmed", (True, False)),
    ("both_fail", (False, False)),
)
CSV_KEYS = ("cell", "arm", "step", "pool", "group")
TOLERANCE = 1e-10
CD_CELLS = (("C", "cross_only"), ("D", "both"))
LOCK_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
WINDOWS_RESERVED_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})
REQUIRED_PREFLIGHT_CHECKS = (
    "initialized_saved_checkpoint_audited",
    "supplied_logits_scored",
    "missing_group_rejected",
    "corrupted_generation_logits_scores_rejected",
    "self_exposure_divergence_verified",
    "frozen_ffn_weight_and_adam_corruption_rejected",
    "unauthorized_run_rejected",
)
EXECUTION_CODE_FILES = (
    "experiments/run_copy_decoder_ffn_adaptation.py",
    "experiments/audit_copy_decoder_ffn_adaptation.py",
    "experiments/validate_copy_exposure_continuation.py",
    "tests/safe_runner.py",
)


def validate_lock_name(lock_name):
    if not isinstance(lock_name, str) or not LOCK_NAME_RE.fullmatch(lock_name):
        raise ValueError("Lock name must be one alphanumeric, underscore or hyphen component.")
    if lock_name.casefold() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Reserved OS device name cannot be used: {lock_name}")


def collect_runtime_environment():
    """Capture live Python/PyTorch/OS/determinism environment metadata."""
    return dict(
        python=sys.version,
        torch=torch.__version__,
        platform=platform.platform(),
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        dtype="float64",
        device="cpu",
    )


def derive_locked_schedule(lock, expected_counts=COUNTS):
    """Derive and cross-check stages, update/example/target budgets, and pools from a verified lock."""
    budget = lock.get("budget")
    gates = lock.get("gates")
    require(isinstance(budget, dict) and isinstance(gates, dict), "Lock missing budget or gates")

    stages_raw = budget.get("stages")
    require(isinstance(stages_raw, list) and len(stages_raw) >= 1 and all(type(s) is int for s in stages_raw),
            "Lock stages must be a non-empty list of ints")
    stages = tuple(stages_raw)
    stage_updates = stages[0] - 6000
    require(stage_updates > 0, "Lock first stage must exceed 6000")
    prev_step = 6000
    for step in stages:
        require(step - prev_step == stage_updates, "Lock stages must have uniform positive spacing")
        prev_step = step

    require(budget.get("window") == [6001, stages[-1]], "Lock budget window mismatch")
    require(budget.get("observed_checkpoints") == [6000, *stages], "Lock observed_checkpoints mismatch")
    require(type(budget.get("decision_step")) is int and budget.get("decision_step") == stages[-1],
            "Lock decision_step mismatch")

    updates_per_cell = budget.get("updates_per_cell")
    examples_per_cell = budget.get("examples_per_cell")
    targets_per_cell = budget.get("targets_per_cell")
    targets_per_update = budget.get("targets_per_update")
    for name, val in (
        ("updates_per_cell", updates_per_cell),
        ("examples_per_cell", examples_per_cell),
        ("targets_per_cell", targets_per_cell),
        ("targets_per_update", targets_per_update),
    ):
        require(type(val) is int and val > 0, f"Lock budget {name} must be positive int")

    n_stages = len(stages)
    require(updates_per_cell == n_stages * stage_updates, "Lock updates_per_cell mismatch")
    require(examples_per_cell % n_stages == 0 and targets_per_cell % n_stages == 0,
            "Lock cell totals must divide evenly across stages")
    stage_examples = examples_per_cell // n_stages
    stage_targets = targets_per_cell // n_stages
    require(stage_targets == stage_updates * targets_per_update, "Lock targets_per_update mismatch")
    require(budget.get("quartet_max_updates") == 4 * updates_per_cell
            and budget.get("quartet_max_examples") == 4 * examples_per_cell
            and budget.get("quartet_max_targets") == 4 * targets_per_cell,
            "Lock quartet budget maxima mismatch")

    short_pools = [p for p in expected_counts if int(p.rsplit("_", 1)[1]) <= 6]
    require(isinstance(gates.get("short_preservation_pools"), list)
            and set(short_pools) == set(gates["short_preservation_pools"])
            and gates.get("short_preservation_groups") == ["distinct", "repeated"],
            "Lock short_preservation pool/group specification mismatch")
    require(type(gates.get("test10_sequences")) is int
            and gates["test10_sequences"] == sum(expected_counts["test_10"]),
            "Lock test10_sequences mismatch")
    return dict(
        stages=stages,
        stage_updates=stage_updates,
        stage_examples=stage_examples,
        stage_targets=stage_targets,
        counts=dict(expected_counts),
    )


def validate_data_pools(data_pools, expected_counts=COUNTS):
    """Validate cohort pool keys, lengths, distinct/repeated counts, and train split order."""
    require(isinstance(data_pools, dict) and set(data_pools) == set(expected_counts),
            "Data pools differ from expected_counts")
    for pool_name, sizes in expected_counts.items():
        rows = data_pools[pool_name]
        length = int(pool_name.rsplit("_", 1)[1])
        require(isinstance(rows, list) and len(rows) == sum(sizes),
                f"Data pool size mismatch at {pool_name}")
        for row in rows:
            require(isinstance(row, list) and len(row) == length and all(type(tok) is int for tok in row),
                    f"Malformed sequence row in {pool_name}")
        n_dist = sum(len(set(row)) == length for row in rows)
        n_rep = len(rows) - n_dist
        require((n_dist, n_rep) == sizes, f"Data pool group composition mismatch at {pool_name}: {(n_dist, n_rep)} != {sizes}")
        if pool_name.startswith("train_"):
            require(sizes[0] == sizes[1]
                    and all(len(set(row)) == length for row in rows[: sizes[0]])
                    and all(len(set(row)) < length for row in rows[sizes[0] :]),
                    f"Training pool half ordering mismatch at {pool_name}")


def _number(value, context):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite or nonnumeric value: {context}")
    return value


def _close_scalar(a, b, context, tol=TOLERANCE):
    require(abs(_number(a, context) - _number(b, context)) <= tol, f"Numeric mismatch: {context}")


def _strict_close(actual, expected, context, tol=1e-9):
    """Compare decoded JSON structures with strict scalar type checks (never conflating bool/int/float)."""
    require(type(actual) is type(expected), f"Type mismatch at {context}: {type(actual).__name__} != {type(expected).__name__}")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"Key mismatch at {context}: {set(actual)} != {set(expected)}")
        for key in expected:
            _strict_close(actual[key], expected[key], f"{context}.{key}", tol=tol)
        return
    if isinstance(expected, list):
        require(len(actual) == len(expected), f"Length mismatch at {context}: {len(actual)} != {len(expected)}")
        for idx, (act_v, exp_v) in enumerate(zip(actual, expected)):
            _strict_close(act_v, exp_v, f"{context}[{idx}]", tol=tol)
        return
    if isinstance(expected, float):
        err = abs(actual - expected)
        require(math.isfinite(err) and err <= tol, f"Float mismatch at {context}: {actual} vs {expected}")
        return
    require(actual == expected, f"Value mismatch at {context}: {actual!r} != {expected!r}")


def _validate_out_dir_chain(out_dir, root_resolved, root_orig=None):
    """Enforce containment under root and non-reparse directory checks along every component."""
    if ".." in out_dir.parts:
        raise ValueError("Output directory cannot contain parent traversal.")
    if not out_dir.is_absolute() and not out_dir.drive and not out_dir.root:
        candidate = root_resolved / out_dir
        parts = candidate.relative_to(root_resolved).parts
    else:
        try:
            parts = out_dir.relative_to(root_resolved).parts
        except ValueError:
            parts = None
            if root_orig is not None:
                root_lex = Path(root_orig)
                try:
                    if root_lex.resolve(strict=True) == root_resolved:
                        parts = out_dir.relative_to(root_lex).parts
                except (OSError, ValueError):
                    parts = None
            if parts is None:
                raise ValueError("Output directory must stay inside checkout root.") from None
    require(len(parts) >= 1, "Output directory cannot be checkout root itself")
    checked_directory(root_resolved)
    for i in range(1, len(parts) + 1):
        checked_directory(root_resolved.joinpath(*parts[:i]))
    return root_resolved.joinpath(*parts).resolve(strict=True)


def _validate_contained_file(out_dir_resolved, root_resolved, rel_str, context):
    """Require relative artifact path to be regular, unredirected, and strictly inside out_dir."""
    require(isinstance(rel_str, str) and bool(rel_str), f"Artifact path must be non-empty string: {context}")
    require("\\" not in rel_str and not rel_str.startswith("/"),
            f"Artifact path escapes containment ({context}): {rel_str}")
    rel_path = Path(rel_str)
    require(not rel_path.is_absolute() and not rel_path.drive and not rel_path.root and ".." not in rel_path.parts,
            f"Artifact path escapes containment ({context}): {rel_str}")
    target = root_resolved / rel_path
    try:
        parts = target.relative_to(root_resolved).parts
    except ValueError:
        raise ValueError(f"Artifact path outside checkout ({context}): {rel_str}") from None
    for i in range(1, len(parts)):
        checked_directory(root_resolved.joinpath(*parts[:i]))
    info = target.lstat()
    require(not redirected(info), f"Linked/reparse-point artifact not supported ({context}): {rel_str}")
    require(stat.S_ISREG(info.st_mode), f"Artifact must be regular file ({context}): {rel_str}")
    resolved = target.resolve(strict=True)
    try:
        resolved.relative_to(out_dir_resolved)
    except ValueError:
        raise ValueError(f"Artifact path must reside inside output directory ({context}): {rel_str}") from None
    return target


def validate_metrics_mapping(metrics, expected_counts=COUNTS):
    """Validate all pool/group schemas, sequence/target counts, bounds, and EOS partition."""
    require(isinstance(metrics, dict), "Metrics must be a JSON object")
    require(set(metrics) == set(expected_counts), "Missing or unexpected evaluation pools")
    for pool, sizes in expected_counts.items():
        groups = metrics[pool]
        require(isinstance(groups, dict) and set(groups) == set(GROUPS), f"Missing or unexpected group: {pool}")
        length = int(pool.rsplit("_", 1)[1])
        for group, count in zip(GROUPS, (sum(sizes), *sizes)):
            values = groups[group]
            if count == 0:
                require(values is None, f"Absent group must be null: {pool}/{group}")
                continue
            require(isinstance(values, dict) and set(values) == set(METRICS), f"Wrong metric fields: {pool}/{group}")
            for key in METRICS:
                value = _number(values[key], f"{pool}/{group}/{key}")
                if key in ("sequences", "targets"):
                    expected = count if key == "sequences" else count * (length + 1)
                    require(type(value) is int and value == expected, f"Wrong count: {pool}/{group}/{key}")
                elif key == "loss":
                    require(value >= 0.0, f"Negative loss: {pool}/{group}")
                else:
                    require(0.0 <= value <= 1.0, f"Metric rate outside [0, 1]: {pool}/{group}/{key}")
            eos_sum = sum(values[k] for k in ("early_eos", "on_time_eos", "late_eos", "missing_eos"))
            _close_scalar(eos_sum, 1.0, f"{pool}/{group}/EOS partition")
        if pool == "test_10" and sizes[0] == 0:
            for key in METRICS:
                _close_scalar(groups["all"][key], groups["repeated"][key], f"test_10 all/repeated {key}")


def compute_historical_ab_gates(ab_report, gates_spec, expected_counts, expected_stages):
    """Recompute historical step-final repair gates from the A/B continuation report."""
    final_step = expected_stages[-1]
    short_pools = [p for p in expected_counts if int(p.rsplit("_", 1)[1]) <= 6]
    ab_runs = {(r.get("arm"), r.get("step")): r for r in ab_report["runs"]}
    own_gates = {}
    for arm in ("cross_only", "both"):
        require((arm, final_step) in ab_runs, f"Missing final historical A/B run for {arm}@{final_step}")
        ms = ab_runs[(arm, final_step)]["evaluation"]["metrics"]
        short_ok = all(
            ms[pool][grp]["exact_accuracy"] >= gates_spec["short_preservation_min_accuracy"]
            for pool in short_pools
            for grp in ("distinct", "repeated")
        )
        long_ok = all(
            ms[pool][grp][metric] >= gates_spec["historical_repair_threshold"]
            for pool, grp in (("test_8", "distinct"), ("test_8", "repeated"), ("test_10", "repeated"))
            for metric in ("exact_accuracy", "on_time_eos")
        )
        own_gates[arm] = dict(short_preserved=short_ok, long_recovered=long_ok, repair=short_ok and long_ok)
    return own_gates


def validate_ab_report_contract(
    root,
    ab_report,
    gates_spec,
    expected_counts,
    expected_stages,
    expected_stage_updates,
    expected_stage_examples,
    expected_stage_targets,
    expected_hist_hashes=None,
):
    """Validate historical A/B report structure, stage accounting, scores/checkpoint hashes, metrics, and gates."""
    require(isinstance(ab_report, dict) and ab_report.get("status") == "complete",
            "Historical A/B report is not complete")
    init = ab_report.get("initial")
    require(isinstance(init, dict), "Historical A/B report missing initial section")
    init_rel = _norm_rel(init.get("scores"))
    require(digest(root / init_rel) == init.get("scores_sha256"),
            "Historical A/B initial scores SHA-256 mismatch")
    if expected_hist_hashes is not None:
        require(init_rel in expected_hist_hashes
                and expected_hist_hashes[init_rel] == init.get("scores_sha256"),
                f"Historical A/B initial scores hash not locked: {init_rel}")
    validate_metrics_mapping(init.get("metrics"), expected_counts=expected_counts)

    ab_runs = ab_report.get("runs")
    expected_keys = {(arm, stop) for arm in ("cross_only", "both") for stop in expected_stages}
    require(isinstance(ab_runs, list) and len(ab_runs) == len(expected_keys),
            "Historical A/B runs do not match expected stage schedule")
    seen_keys = set()
    for ab_run in ab_runs:
        require(isinstance(ab_run, dict), "Historical A/B run entry must be an object")
        key = (ab_run.get("arm"), ab_run.get("step"))
        require(key in expected_keys and key not in seen_keys,
                f"Incompatible or duplicate historical A/B stage run: {key}")
        seen_keys.add(key)
        require(ab_run.get("updates") == expected_stage_updates
                and ab_run.get("examples") == expected_stage_examples
                and ab_run.get("active_targets") == expected_stage_targets,
                f"Historical A/B segment counts mismatch at {key}")
        ev = ab_run.get("evaluation")
        require(isinstance(ev, dict), f"Historical A/B run missing evaluation at {key}")
        scores_rel = _norm_rel(ev.get("scores"))
        require(digest(root / scores_rel) == ev.get("scores_sha256"),
                f"Historical A/B scores SHA-256 mismatch: {scores_rel}")
        validate_metrics_mapping(ev.get("metrics"), expected_counts=expected_counts)
        ckpt_rel = _norm_rel(ab_run.get("checkpoint"))
        ckpt_sha = ab_run.get("checkpoint_sha256")
        if expected_hist_hashes is not None:
            require(ckpt_rel in expected_hist_hashes and expected_hist_hashes[ckpt_rel] == ckpt_sha,
                    f"Historical A/B checkpoint hash not locked: {ckpt_rel}")
            require(scores_rel in expected_hist_hashes and expected_hist_hashes[scores_rel] == ev.get("scores_sha256"),
                    f"Historical A/B scores hash not locked: {scores_rel}")

    ab_gates = ab_report.get("gates")
    require(isinstance(ab_gates, dict) and set(ab_gates) == {"cross_only", "both"},
            "Historical A/B report missing gates metadata")
    expected_gates = compute_historical_ab_gates(ab_report, gates_spec, expected_counts, expected_stages)
    for arm in ("cross_only", "both"):
        require(isinstance(ab_gates[arm], dict)
                and ab_gates[arm] == expected_gates[arm]
                and all(type(v) is bool for v in ab_gates[arm].values()),
                f"Historical A/B report gates mismatch at {arm}")


def audit_supplied_cell(model, rows, arm, stored, expected, pos=None):
    """Non-strippable wrapper exercising reference evaluator, own_measures, and independent audit_cell."""
    require_debug_assertions()
    require(isinstance(rows, list) and len(rows) > 0, "Pool rows must be a non-empty list")
    n_seq = len(rows)
    n_tok = len(rows[0])
    require(isinstance(stored, dict) and {"logits", "scores", "generated"}.issubset(stored),
            "Stored score payload missing required keys")
    logits = stored["logits"]
    scores = stored["scores"]
    generated = stored["generated"]
    require(torch.is_tensor(logits) and logits.dtype == torch.float64 and logits.shape == (n_seq, n_tok + 1, 16),
            f"Unexpected stored logits dtype or shape: {getattr(logits, 'dtype', None)} {getattr(logits, 'shape', None)}")
    require(torch.is_tensor(scores) and scores.dtype == torch.float64 and scores.shape == (n_seq, 2, n_tok + 1),
            f"Unexpected stored scores dtype or shape: {getattr(scores, 'dtype', None)} {getattr(scores, 'shape', None)}")
    require(isinstance(generated, list) and len(generated) == n_seq, "Stored generated length mismatch")

    err, metric_err = audit_cell(model, rows, pos, arm, stored, expected)
    require(isinstance(err, float) and math.isfinite(err) and err <= 1e-9,
            f"Reference logit/score error out of tolerance: {err}")
    require(isinstance(metric_err, float) and math.isfinite(metric_err) and metric_err <= 1e-9,
            f"Reference metric error out of tolerance: {metric_err}")
    own_m = own_measures(rows, logits, scores, generated)
    _strict_close(own_m, expected, "audit_supplied_cell.own_measures", tol=1e-9)
    return err, metric_err


def independent_replay_streams(train_pools, gen, pg, begin_step, stop_step):
    """Independently replay RNG draws, indices, and position layouts without producer sample helpers."""
    stream = hashlib.sha256()
    pstream = hashlib.sha256()
    examples = 0
    targets_count = 0
    for step in range(begin_step + 1, stop_step + 1):
        n = 3 + (step - 1) % 4
        batch = 840 // (n + 1)
        half = len(train_pools[f"train_{n}"]) // 2
        ranks = torch.rand(2, 105, generator=gen, dtype=torch.float64)
        idx = torch.floor(ranks[:, : batch // 2] * half).long()
        idx[1] += half
        idx_list = idx.flatten().tolist()
        pos_list = torch.sort(torch.randperm(11, generator=pg)[: n + 1]).values.tolist()
        stream.update(json.dumps([n, idx_list]).encode())
        pstream.update(json.dumps(pos_list).encode())
        examples += batch
        targets_count += batch * (n + 1)
    return stream.hexdigest(), pstream.hexdigest(), examples, targets_count


def _verify_adam_group(group):
    require(group.get("lr") == 0.001, f"Unexpected Adam lr: {group.get('lr')}")
    betas = group.get("betas")
    require(isinstance(betas, (list, tuple)) and list(betas) == [0.9, 0.999], f"Unexpected Adam betas: {betas}")
    require(group.get("eps") == 1e-8, f"Unexpected Adam eps: {group.get('eps')}")
    require(type(group.get("weight_decay")) is not bool and group.get("weight_decay") == 0,
            f"Unexpected Adam weight_decay: {group.get('weight_decay')}")
    require(group.get("amsgrad") is False, f"Unexpected Adam amsgrad: {group.get('amsgrad')}")
    require(group.get("maximize", False) is False, f"Unexpected Adam maximize: {group.get('maximize')}")
    require(group.get("decoupled_weight_decay", False) is False,
            f"Unexpected Adam decoupled_weight_decay: {group.get('decoupled_weight_decay')}")


def capture_ffn_snapshot(checkpoint):
    """Capture deep-copied frozen FFN parameter tensors and Adam state directly by parameter ID."""
    probe = make_model("rope")
    named = tuple(probe.named_parameters())
    param_count = len(named)
    groups = checkpoint["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["params"] == list(range(param_count)),
            "Unexpected optimizer param group contract")
    _verify_adam_group(groups[0])
    opt_state = checkpoint["optimizer"]["state"]
    require(set(opt_state.keys()) == set(range(param_count)), "Incomplete optimizer state table")
    ffn_ids = {name: idx for idx, (name, _) in enumerate(named) if name in FFN_NAMES}
    require(set(ffn_ids) == set(FFN_NAMES), "Missing FFN parameter name in model")
    return {
        name: dict(
            param_id=pid,
            tensor=checkpoint["model"][name].detach().clone(),
            adam=copy.deepcopy(opt_state[pid]),
        )
        for name, pid in ffn_ids.items()
    }, param_count


def verify_frozen_checkpoint(cp, prior_cp, start_ffn, expected_step, expected_start_step=6000):
    """Verify frozen FFN weights/Adam moments stayed bitwise identical while non-FFN params trained."""
    require(isinstance(cp, dict)
            and set(cp.keys()) == {"model", "optimizer", "step", "generator", "position_generator"},
            "Unexpected checkpoint keys")
    require(type(cp["step"]) is int and cp["step"] == expected_step,
            f"Unexpected checkpoint step {cp['step']} != {expected_step}")

    probe = make_model("rope")
    named = tuple(probe.named_parameters())
    param_count = len(named)
    require(set(cp["model"]) == {name for name, _ in named}, "Unexpected or missing model state keys")
    groups = cp["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["params"] == list(range(param_count)),
            "Optimizer parameter ordering violated")
    _verify_adam_group(groups[0])
    opt_state = cp["optimizer"]["state"]
    require(set(opt_state.keys()) == set(range(param_count)), "Optimizer state keys mismatch")

    trainable_moved = False
    for pid, (name, probe_param) in enumerate(named):
        tensor = cp["model"][name]
        require(torch.is_tensor(tensor) and tensor.dtype == torch.float64 and tensor.shape == probe_param.shape
                and bool(torch.isfinite(tensor).all()),
                f"Invalid checkpoint tensor at {name}")
        state = opt_state[pid]
        require(isinstance(state, dict) and {"step", "exp_avg", "exp_avg_sq"}.issubset(state),
                f"Incomplete Adam state keys for {name}")
        for mkey in ("exp_avg", "exp_avg_sq"):
            mtensor = state[mkey]
            require(torch.is_tensor(mtensor) and mtensor.dtype == torch.float64 and mtensor.shape == probe_param.shape
                    and bool(torch.isfinite(mtensor).all()),
                    f"Invalid Adam {mkey} tensor for {name}")
        step_val = float(state["step"])
        require(math.isfinite(step_val) and step_val.is_integer(), f"Noninteger Adam step for {name}: {state['step']}")
        if name in FFN_NAMES:
            snap = start_ffn[name]
            require(same(tensor, snap["tensor"]),
                    f"Frozen decoder FFN parameter moved: {name}")
            require(same(state, snap["adam"]),
                    f"Frozen decoder FFN Adam state moved: {name}")
            require(int(step_val) == expected_start_step,
                    f"Frozen decoder FFN Adam step advanced: {name}")
        else:
            require(int(step_val) == expected_step,
                    f"Trainable parameter Adam step mismatch: {name}")
            if not same(tensor, prior_cp["model"][name]):
                trainable_moved = True
    require(trainable_moved, f"No trainable parameter changed up to step {expected_step}")
    model = make_model("rope")
    model.load_state_dict(cp["model"], strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    opt.load_state_dict(cp["optimizer"])
    check_order(model, opt)
    return model


def _paired_counts(rows, seqs_first, seqs_second):
    require(len(rows) == len(seqs_first) == len(seqs_second), "Paired sequence count mismatch")
    counts = {label: 0 for label, _ in PAIR_LABELS}
    first_ok = 0
    second_ok = 0
    for row, a_out, b_out in zip(rows, seqs_first, seqs_second):
        target = row + [2]
        a_hit = a_out == target
        b_hit = b_out == target
        first_ok += int(a_hit)
        second_ok += int(b_hit)
        for label, key in PAIR_LABELS:
            if (a_hit, b_hit) == key:
                counts[label] += 1
                break
    total = len(rows)
    return dict(
        sequences=total,
        first_correct=first_ok,
        second_correct=second_ok,
        **counts,
        net_gain=(second_ok - first_ok) / total if total else 0.0,
    )


def audit_csv(csv_text, cd_runs):
    """Independently parse metrics.csv and reconcile every scalar metric with report runs."""
    reader = csv.DictReader(csv_text.splitlines(keepends=True), strict=True)
    fields = reader.fieldnames or []
    require(len(fields) == len(set(fields)) and set(fields) == set(CSV_KEYS + METRICS),
            "Missing, duplicate, or unexpected metrics.csv columns")
    expected = {}
    for run in cd_runs:
        for pool, groups in run["evaluation"]["metrics"].items():
            for group, values in groups.items():
                if values is not None:
                    expected[(run["cell"], run["arm"], run["step"], pool, group)] = values

    observed = set()
    for row in reader:
        require(set(row) == set(fields) and all(v is not None for v in row.values()), "Ragged CSV row")
        require(re.fullmatch(r"[0-9]+", row["step"]) is not None, "Invalid CSV step")
        step = int(row["step"])
        require(row["step"] == str(step), "Noncanonical CSV step")
        key = (row["cell"], row["arm"], step, row["pool"], row["group"])
        require(key not in observed and key in expected, f"Unexpected or duplicate CSV key: {key}")
        observed.add(key)
        for field in METRICS:
            text = row[field]
            if field in ("sequences", "targets"):
                require(re.fullmatch(r"[0-9]+", text) is not None, f"Invalid integer CSV metric {field}")
                require(int(text) == expected[key][field], f"CSV count mismatch at {key}/{field}")
            else:
                val = _number(float(text), f"{key}/{field}")
                require(val >= 0.0 and (field == "loss" or val <= 1.0), f"CSV value out of range at {key}/{field}")
                _close_scalar(val, expected[key][field], f"CSV {key}/{field}")
    require(observed == set(expected), f"Missing CSV records: {sorted(set(expected) - observed)}")
    return len(observed)


def _recompute_decision(root, ab_report, cd_report, gates_spec, stages, expected_counts, data_pools):
    final_step = stages[-1]
    ab_metrics = {
        (arm, final_step): next(r for r in ab_report["runs"] if r["arm"] == arm and r["step"] == final_step)
        for arm in ("cross_only", "both")
    }
    cd_metrics = {
        (cell, final_step): next(r for r in cd_report["runs"] if r["cell"] == cell and r["step"] == final_step)
        for cell, _ in CD_CELLS
    }
    cell_runs = {
        "A": ab_metrics[("cross_only", final_step)],
        "B": ab_metrics[("both", final_step)],
        "C": cd_metrics[("C", final_step)],
        "D": cd_metrics[("D", final_step)],
    }

    acc10 = {c: r["evaluation"]["metrics"]["test_10"]["all"]["exact_accuracy"] for c, r in cell_runs.items()}
    ab_effect = acc10["B"] - acc10["A"]
    dc_effect = acc10["D"] - acc10["C"]
    interaction = ab_effect - dc_effect

    scores10 = {
        c: torch.load(root / _norm_rel(r["evaluation"]["scores"]), weights_only=True)["test_10"]["generated"]
        for c, r in cell_runs.items()
    }
    rows10 = data_pools["test_10"]
    paired10 = {
        f"{x}_vs_{y}": _paired_counts(rows10, scores10[x], scores10[y])
        for x, y in (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"))
    }

    short_pools = [p for p in expected_counts if int(p.rsplit("_", 1)[1]) <= 6]
    if "short_preservation_pools" in gates_spec:
        require(set(short_pools) == set(gates_spec["short_preservation_pools"]),
                "Short preservation pools mismatch")
    all_checkpoints = [6000, *stages]
    ab_by_key = {(r["arm"], r["step"]): r["evaluation"]["metrics"] for r in ab_report["runs"]}
    cd_by_key = {(r["cell"], r["step"]): r["evaluation"]["metrics"] for r in cd_report["runs"]}

    short_failures = []
    short_passed_by_cell = {c: True for c in ("A", "B", "C", "D")}
    worst_short = 1.0
    for step in all_checkpoints:
        step_metrics = (
            {
                "A": ab_report["initial"]["metrics"],
                "B": ab_report["initial"]["metrics"],
                "C": cd_report["initial"]["metrics"],
                "D": cd_report["initial"]["metrics"],
            }
            if step == 6000
            else {"A": ab_by_key[("cross_only", step)], "B": ab_by_key[("both", step)],
                  "C": cd_by_key[("C", step)], "D": cd_by_key[("D", step)]}
        )
        for cell, metrics in step_metrics.items():
            for pool in short_pools:
                for group in ("distinct", "repeated"):
                    acc = metrics[pool][group]["exact_accuracy"]
                    worst_short = min(worst_short, acc)
                    if acc < gates_spec["short_preservation_min_accuracy"]:
                        short_passed_by_cell[cell] = False
                        short_failures.append(dict(cell=cell, step=step, pool=pool, group=group, exact_accuracy=acc))

    final_short_ok = {
        c: all(
            cell_runs[c]["evaluation"]["metrics"][pool][group]["exact_accuracy"]
            >= gates_spec["short_preservation_min_accuracy"]
            for pool in short_pools
            for group in ("distinct", "repeated")
        )
        for c in ("A", "B", "C", "D")
    }

    scores8 = {
        c: torch.load(root / _norm_rel(r["evaluation"]["scores"]), weights_only=True)["test_8"]["generated"]
        for c, r in cell_runs.items()
    }
    rows8 = data_pools["test_8"]
    len8_groups = {}
    for group in ("distinct", "repeated", "all"):
        ids = [i for i, row in enumerate(rows8) if group == "all" or (len(set(row)) == len(row)) == (group == "distinct")]
        group_rows = [rows8[i] for i in ids]
        g_acc = {c: r["evaluation"]["metrics"]["test_8"][group]["exact_accuracy"] for c, r in cell_runs.items()}
        len8_groups[group] = dict(
            accuracy=g_acc,
            b_minus_a=g_acc["B"] - g_acc["A"],
            d_minus_c=g_acc["D"] - g_acc["C"],
            interaction=(g_acc["B"] - g_acc["A"]) - (g_acc["D"] - g_acc["C"]),
            paired={
                f"{x}_vs_{y}": _paired_counts(group_rows, [scores8[x][i] for i in ids], [scores8[y][i] for i in ids])
                for x, y in (("A", "B"), ("C", "D"))
            },
        )

    repair = {}
    threshold = gates_spec["historical_repair_threshold"]
    for cell, run in cell_runs.items():
        ms = run["evaluation"]["metrics"]
        long_ok = all(
            ms[pool][grp][metric] >= threshold
            for pool, grp in (("test_8", "distinct"), ("test_8", "repeated"), ("test_10", "repeated"))
            for metric in ("exact_accuracy", "on_time_eos")
        )
        repair[cell] = final_short_ok[cell] and long_ok

    contrast_ok = interaction >= gates_spec["material_attenuation_min_gain"] and ab_effect > 0.0
    return dict(
        decision_step=final_step,
        primary=dict(
            accuracy=acc10,
            b_minus_a=ab_effect,
            d_minus_c=dc_effect,
            interaction=interaction,
            material_attenuation_min_gain=gates_spec["material_attenuation_min_gain"],
            contrast_threshold_met=contrast_ok,
            material_attenuation=contrast_ok and (not short_failures),
            positive_exposure_benefit=ab_effect > 0.0,
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
        secondary_length8=len8_groups,
        historical_repair_passed=repair,
    )


def _validate_preflight_json(pre):
    require(isinstance(pre, dict), "preflight.json must be an object")
    require(pre.get("passed") is True and pre.get("formal_ready") is False
            and pre.get("formal_authorized") is False and type(pre.get("formal_updates")) is int
            and pre.get("formal_updates") == 0 and pre.get("auditor_reloaded") is True,
            "Invalid preflight flags in preflight.json")
    checks = pre.get("checks")
    require(isinstance(checks, dict) and set(checks) == set(REQUIRED_PREFLIGHT_CHECKS)
            and all(v is True for v in checks.values()),
            "Incomplete preflight checks in preflight.json")
    grad_checks = pre.get("gradient_checks")
    require(isinstance(grad_checks, list) and len(grad_checks) == 2
            and all(isinstance(g, dict) and g.get("passed") is True for g in grad_checks),
            "Incomplete gradient_checks in preflight.json")


@dataclass(frozen=True)
class _AuditOverrides:
    ab_report_rel: str = f"{CONTINUATION_REL}/report.json"
    expected_counts: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(COUNTS))
    expected_stages: tuple[int, ...] = STAGES
    expected_stage_updates: int = STAGE_UPDATES
    expected_stage_examples: int = STAGE_EXAMPLES
    expected_stage_targets: int = STAGE_TARGETS
    verify_lock: bool = True
    check_ab_checkpoints: bool = True


def _validate_fixture_schedule(expected_counts, expected_stages, expected_stage_updates,
                             expected_stage_examples, expected_stage_targets):
    require(isinstance(expected_counts, dict) and len(expected_counts) > 0, "Invalid expected_counts")
    require(len(expected_stages) >= 1 and all(type(s) is int for s in expected_stages),
            "Invalid expected_stages")
    require(type(expected_stage_updates) is int and expected_stage_updates > 0,
            "Invalid expected_stage_updates")
    require(type(expected_stage_examples) is int and expected_stage_examples > 0,
            "Invalid expected_stage_examples")
    require(type(expected_stage_targets) is int and expected_stage_targets > 0,
            "Invalid expected_stage_targets")
    prev_step = 6000
    for step in expected_stages:
        require(step - prev_step == expected_stage_updates, "Nonuniform stage increment")
        prev_step = step


def _audit_directory_impl(
    directory,
    root=ROOT,
    lock_name=DEFAULT_LOCK_NAME,
    save=True,
    diagnostic_cross_runtime=False,
    overrides=None,
):
    """Independently verify lock, runtime, preflight, checkpoints, frozen FFN state, streams, evaluations, CSV, and gates."""
    require_debug_assertions()
    validate_lock_name(lock_name)
    root_orig = root
    root = Path(root_orig).resolve(strict=True)
    out_dir = _validate_out_dir_chain(Path(directory), root, root_orig=root_orig)
    if diagnostic_cross_runtime:
        require(not save, "Diagnostic cross-runtime audit cannot save formal audit.json; pass save=False (--no-save)")
    if save:
        require(not (out_dir / "audit.json").exists(), "Refusing to overwrite existing audit.json")

    verify_lock = True if overrides is None else overrides.verify_lock
    check_ab_checkpoints = True if overrides is None else overrides.check_ab_checkpoints
    ab_report_rel = f"{CONTINUATION_REL}/report.json" if overrides is None else overrides.ab_report_rel

    if verify_lock:
        verify_lock_bundle(root=root, lock_name=lock_name)
    lock_dir = root / "results" / lock_name
    checked_directory(lock_dir)
    lock = decode_json((lock_dir / "lock.json").read_text(encoding="utf-8"))
    require(isinstance(lock, dict), "lock.json must be an object")
    require(list(FFN_NAMES) == lock["budget"]["frozen_parameters"],
            "Model FFN_NAMES differs from locked frozen_parameters")

    if overrides is None:
        sched = derive_locked_schedule(lock, COUNTS)
        expected_counts = sched["counts"]
        expected_stages = sched["stages"]
        expected_stage_updates = sched["stage_updates"]
        expected_stage_examples = sched["stage_examples"]
        expected_stage_targets = sched["stage_targets"]
    else:
        expected_counts = overrides.expected_counts
        expected_stages = tuple(overrides.expected_stages)
        expected_stage_updates = overrides.expected_stage_updates
        expected_stage_examples = overrides.expected_stage_examples
        expected_stage_targets = overrides.expected_stage_targets
        _validate_fixture_schedule(
            expected_counts, expected_stages, expected_stage_updates,
            expected_stage_examples, expected_stage_targets,
        )

    audit_env = collect_runtime_environment()
    require(type(audit_env["threads"]) is int and audit_env["threads"] == lock["budget"]["threads"]
            and audit_env["deterministic_algorithms"] is lock["budget"]["deterministic_algorithms"],
            "Auditor live threads or deterministic_algorithms setting differs from locked budget")
    runtime_matches = (audit_env["torch"] == lock["environment"]["torch"])
    if not diagnostic_cross_runtime:
        require(runtime_matches,
                f"Auditor torch {audit_env['torch']} != locked torch {lock['environment']['torch']}")

    for name in ("plan.json", "report.json", "preflight.json", "metrics.csv"):
        _validate_contained_file(out_dir, root, _norm_rel(str(out_dir.relative_to(root) / name)), name)

    plan = decode_json((out_dir / "plan.json").read_text(encoding="utf-8"))
    report = decode_json((out_dir / "report.json").read_text(encoding="utf-8"))
    pre = decode_json((out_dir / "preflight.json").read_text(encoding="utf-8"))
    csv_raw = (out_dir / "metrics.csv").read_bytes()

    require(isinstance(plan, dict) and isinstance(report, dict), "plan.json and report.json must be objects")
    _validate_preflight_json(pre)
    pre_env = pre.get("environment")
    require(isinstance(pre_env, dict), "Missing preflight environment metadata")
    for key in ("threads", "deterministic_algorithms", "dtype", "device"):
        _strict_close(pre_env.get(key), lock["budget"][key], f"preflight.environment.{key}")
    require(report.get("status") == "completed_unaudited",
            f"Unauditable report status: {report.get('status')}")
    require(plan.get("lock_sha256") == digest(lock_dir / "lock.json"), "Plan lock_sha256 mismatch")
    require(report.get("plan_sha256") == digest(out_dir / "plan.json"), "Report plan_sha256 mismatch")
    require(plan.get("preflight_sha256") == digest(out_dir / "preflight.json"), "Plan preflight_sha256 mismatch")
    require(report.get("metrics_sha256") == hashlib.sha256(csv_raw).hexdigest(), "Report metrics_sha256 mismatch")

    require(type(plan.get("start")) is int and plan.get("start") == 6000, "Plan start must be 6000")
    require(plan.get("stages") == list(expected_stages), "Plan stages mismatch")
    require(plan.get("cells") == {"C": lock["budget"]["cells"]["C"], "D": lock["budget"]["cells"]["D"]},
            "Plan cells differ from locked budget")
    require(plan.get("frozen_parameters") == list(FFN_NAMES), "Plan frozen_parameters mismatch")

    require(isinstance(plan.get("execution_code_hashes"), dict)
            and set(plan["execution_code_hashes"]) == set(EXECUTION_CODE_FILES),
            "Missing execution_code_hashes in plan.json")
    verify_protected_hashes(root, plan["execution_code_hashes"])
    verify_protected_hashes(root, lock["study_hashes"])
    verify_protected_hashes(root, lock["historical_artifact_hashes"])
    require(plan.get("protected_hashes") == lock["protected_hashes"], "Protected hash manifest drifted from lock")
    verify_protected_hashes(root, plan["protected_hashes"])

    data_raw = json.loads((root / DATA_REL).read_text(encoding="utf-8"))
    data_pools = {k: v for k, v in data_raw["pools"].items() if not k.startswith("uniform_")}
    require(digest(root / DATA_REL) == lock["historical_artifact_hashes"][DATA_REL], "Data SHA-256 mismatch")
    validate_data_pools(data_pools, expected_counts=expected_counts)
    train_pools = {k: torch.tensor(v) for k, v in data_pools.items() if k.startswith("train_")}

    start_path = root / START_REL
    start_sha = digest(start_path)
    require(start_sha == lock["historical_artifact_hashes"][START_REL], "Start checkpoint SHA-256 mismatch")
    start_cp = torch.load(start_path, weights_only=True)
    require(type(start_cp.get("step")) is int and start_cp.get("step") == 6000, "Start checkpoint step must be 6000")
    start_ffn, _ = capture_ffn_snapshot(start_cp)

    ab_report_path = root / _norm_rel(ab_report_rel)
    require(digest(ab_report_path) == lock["historical_artifact_hashes"][_norm_rel(ab_report_rel)],
            "Historical A/B report SHA-256 mismatch")
    ab_report = json.loads(ab_report_path.read_text(encoding="utf-8"))
    require(plan.get("historical_ab_report_sha256") == digest(ab_report_path), "Plan historical A/B report hash mismatch")
    validate_ab_report_contract(
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

    # Verify initial update-6000 evaluation matches historical initial evaluation.
    start_model = make_model("rope").eval()
    start_model.load_state_dict(start_cp["model"], strict=True)
    init_ev = report["initial"]
    init_scores_path = _validate_contained_file(out_dir, root, init_ev.get("scores"), "initial.scores")
    require(digest(init_scores_path) == init_ev["scores_sha256"], "Initial scores SHA-256 mismatch")
    validate_metrics_mapping(init_ev["metrics"], expected_counts=expected_counts)
    init_saved = torch.load(init_scores_path, weights_only=True)
    max_error = 0.0
    max_metric_error = 0.0
    cells_audited = 0
    for pool_name, rows in data_pools.items():
        err, merr = audit_supplied_cell(start_model, rows, "cross_only", init_saved[pool_name], init_ev["metrics"][pool_name])
        max_error = max(max_error, err)
        max_metric_error = max(max_metric_error, merr)
        cells_audited += 1

    hist_init_scores = torch.load(root / _norm_rel(ab_report["initial"]["scores"]), weights_only=True)
    for pool_name in data_pools:
        require(torch.equal(init_saved[pool_name]["logits"], hist_init_scores[pool_name]["logits"])
                and torch.equal(init_saved[pool_name]["scores"], hist_init_scores[pool_name]["scores"])
                and init_saved[pool_name]["generated"] == hist_init_scores[pool_name]["generated"],
                f"Initial evaluation diverges from historical continuation at {pool_name}")

    runs = report.get("runs")
    expected_runs = {(cell, arm, stop) for cell, arm in CD_CELLS for stop in expected_stages}
    require(isinstance(runs, list) and len(runs) == len(expected_runs), "Unexpected number of C/D stage runs")
    runs_by_key = {}
    for run in runs:
        require(isinstance(run, dict), "Run entry must be an object")
        require(run.get("decoder_ffn_frozen") is True, f"Run must record decoder_ffn_frozen=True: {run.get('cell')}")
        for int_key in ("step", "updates", "examples", "active_targets"):
            require(type(run.get(int_key)) is int, f"Run {int_key} must be int: {run.get('cell')}")
        key = (run.get("cell"), run.get("arm"), run.get("step"))
        require(key in expected_runs and key not in runs_by_key, f"Invalid or duplicate C/D run key: {key}")
        runs_by_key[key] = run

    arm_gens = {}
    prior_cps = {}
    prior_shas = {}
    for cell, _ in CD_CELLS:
        g = torch.Generator()
        g.set_state(start_cp["generator"].clone())
        pg = torch.Generator()
        pg.set_state(start_cp["position_generator"].clone())
        arm_gens[cell] = (g, pg)
        prior_cps[cell] = start_cp
        prior_shas[cell] = start_sha

    prev_step = 6000
    for stop in expected_stages:
        stage_runs = []
        ab_stage = {
            arm: next(r for r in ab_report["runs"] if r["arm"] == arm and r["step"] == stop)
            for arm in ("cross_only", "both")
        }
        for cell, arm in CD_CELLS:
            run = runs_by_key[(cell, arm, stop)]
            require(run["start_sha256"] == prior_shas[cell], f"Broken checkpoint lineage at {cell}@{stop}")
            require(run["updates"] == expected_stage_updates and run["examples"] == expected_stage_examples
                    and run["active_targets"] == expected_stage_targets,
                    f"Segment counts mismatch at {cell}@{stop}")
            ckpt_path = _validate_contained_file(out_dir, root, run.get("checkpoint"), f"{cell}@{stop}.checkpoint")
            require(digest(ckpt_path) == run["checkpoint_sha256"], f"Checkpoint SHA-256 mismatch at {cell}@{stop}")
            cp = torch.load(ckpt_path, weights_only=True)
            model = verify_frozen_checkpoint(cp, prior_cps[cell], start_ffn, stop, expected_start_step=6000)

            stream_hex, pstream_hex, ex_count, tgt_count = independent_replay_streams(
                train_pools, arm_gens[cell][0], arm_gens[cell][1], prev_step, stop
            )
            require(ex_count == run["examples"] and tgt_count == run["active_targets"],
                    f"Recomputed segment counts mismatch at {cell}@{stop}")
            require(stream_hex == run["stream"] == ab_stage[arm]["stream"],
                    f"Data stream mismatch against historical A/B at {cell}@{stop}")
            require(pstream_hex == run["position_stream"] == ab_stage[arm]["position_stream"],
                    f"Position stream mismatch against historical A/B at {cell}@{stop}")
            require(torch.equal(cp["generator"], arm_gens[cell][0].get_state())
                    and torch.equal(cp["position_generator"], arm_gens[cell][1].get_state()),
                    f"Saved RNG state mismatch at {cell}@{stop}")
            if check_ab_checkpoints:
                ab_ckpt_path = root / _norm_rel(ab_stage[arm]["checkpoint"])
                require(digest(ab_ckpt_path) == ab_stage[arm]["checkpoint_sha256"],
                        f"Historical A/B checkpoint SHA-256 mismatch: {ab_ckpt_path}")
                ab_cp = torch.load(ab_ckpt_path, weights_only=True)
                require(torch.equal(cp["generator"], ab_cp["generator"])
                        and torch.equal(cp["position_generator"], ab_cp["position_generator"]),
                        f"RNG state diverged from historical checkpoint at {cell}@{stop}")

            ev = run["evaluation"]
            scores_path = _validate_contained_file(out_dir, root, ev.get("scores"), f"{cell}@{stop}.scores")
            require(digest(scores_path) == ev["scores_sha256"], f"Scores SHA-256 mismatch at {cell}@{stop}")
            validate_metrics_mapping(ev["metrics"], expected_counts=expected_counts)
            saved = torch.load(scores_path, weights_only=True)
            model.eval()
            for pool_name, rows in data_pools.items():
                err, merr = audit_supplied_cell(model, rows, "cross_only", saved[pool_name], ev["metrics"][pool_name])
                max_error = max(max_error, err)
                max_metric_error = max(max_metric_error, merr)
                cells_audited += 1

            prior_cps[cell] = cp
            prior_shas[cell] = run["checkpoint_sha256"]
            stage_runs.append(cp)

        require(torch.equal(stage_runs[0]["generator"], stage_runs[1]["generator"])
                and torch.equal(stage_runs[0]["position_generator"], stage_runs[1]["position_generator"]),
                f"C/D generator state mismatch at step {stop}")
        require(not same(stage_runs[0]["model"], stage_runs[1]["model"]),
                f"C and D models failed to diverge at step {stop}")
        prev_step = stop

    cd_updates = sum(r["updates"] for r in runs)
    require(type(report.get("formal_updates")) is int and report.get("formal_updates") == cd_updates,
            "Report formal_updates mismatch")
    total_updates = ab_report_updates(ab_report) + cd_updates
    require(total_updates == 4 * len(expected_stages) * expected_stage_updates,
            "Quartet total update count mismatch")
    rep_env = report.get("environment")
    require(isinstance(rep_env, dict), "Missing report environment metadata")
    for key in ("threads", "deterministic_algorithms", "dtype", "device"):
        _strict_close(rep_env.get(key), lock["budget"][key], f"report.environment.{key}")
    if verify_lock:
        _strict_close(rep_env.get("torch"), lock["environment"]["torch"], "report.environment.torch")

    csv_rows = audit_csv(csv_raw.decode("utf-8"), runs)
    own_decision = _recompute_decision(root, ab_report, report, lock["gates"], expected_stages, expected_counts, data_pools)
    _strict_close(own_decision, report.get("decision"), "decision", tol=1e-9)
    for cell, arm in (("A", "cross_only"), ("B", "both")):
        require(own_decision["historical_repair_passed"][cell] == ab_report["gates"][arm]["repair"],
                f"Quartet historical_repair_passed[{cell}] diverged from historical A/B report gates")

    verify_protected_hashes(root, plan["protected_hashes"])
    verify_protected_hashes(root, plan["execution_code_hashes"])
    verify_protected_hashes(root, lock["study_hashes"])
    verify_protected_hashes(root, lock["historical_artifact_hashes"])

    is_formal = bool(verify_lock and check_ab_checkpoints and runtime_matches and not diagnostic_cross_runtime)
    if diagnostic_cross_runtime or not runtime_matches:
        audit_scope = "diagnostic_cross_runtime"
        limitation = (
            "Cross-runtime diagnostic audit only (live torch differs from locked torch or diagnostic mode enabled); "
            "NOT valid for formal acceptance."
        )
    elif not (verify_lock and check_ab_checkpoints):
        audit_scope = "diagnostic_unverified_lock_or_checkpoints"
        limitation = "Diagnostic non-canonical audit without full lock and A/B checkpoint verification; NOT valid for formal acceptance."
    else:
        audit_scope = "formal"
        limitation = "Saved-checkpoint forward/decoding/metrics/stream/frozen-FFN audit; gradient step equivalence checked by runner contract and tests."

    result = dict(
        passed=True,
        formal_acceptance=is_formal,
        audit_scope=audit_scope,
        runtime_matches_locked_torch=runtime_matches,
        audit_environment=audit_env,
        cells_audited=cells_audited,
        csv_rows=csv_rows,
        max_reference_error=max_error,
        max_metric_error=max_metric_error,
        cd_updates=cd_updates,
        cd_examples=sum(r["examples"] for r in runs),
        cd_targets=sum(r["active_targets"] for r in runs),
        ab_plus_cd_total_updates=total_updates,
        frozen_ffn_bitwise_unchanged=True,
        paired_streams_exact=True,
        decision_reconciled=True,
        protected_files=len(plan["protected_hashes"]),
        report_sha256=digest(out_dir / "report.json"),
        limitation=limitation,
    )
    if save:
        write(out_dir / "audit.json", result)
    return result


def audit_directory(directory, root=ROOT, lock_name=DEFAULT_LOCK_NAME, save=True, diagnostic_cross_runtime=False):
    """Independently audit a formal C/D run strictly against the verified locked schedule and environment."""
    return _audit_directory_impl(
        directory,
        root=root,
        lock_name=lock_name,
        save=save,
        diagnostic_cross_runtime=diagnostic_cross_runtime,
        overrides=None,
    )


def _audit_directory_for_tests(
    directory,
    root=ROOT,
    lock_name=DEFAULT_LOCK_NAME,
    ab_report_rel=f"{CONTINUATION_REL}/report.json",
    expected_counts=COUNTS,
    expected_stages=STAGES,
    expected_stage_updates=STAGE_UPDATES,
    expected_stage_examples=STAGE_EXAMPLES,
    expected_stage_targets=STAGE_TARGETS,
    verify_lock=True,
    check_ab_checkpoints=True,
    save=True,
    diagnostic_cross_runtime=False,
):
    return _audit_directory_impl(
        directory,
        root=root,
        lock_name=lock_name,
        save=save,
        diagnostic_cross_runtime=diagnostic_cross_runtime,
        overrides=_AuditOverrides(
            ab_report_rel=ab_report_rel,
            expected_counts=dict(expected_counts),
            expected_stages=tuple(expected_stages),
            expected_stage_updates=expected_stage_updates,
            expected_stage_examples=expected_stage_examples,
            expected_stage_targets=expected_stage_targets,
            verify_lock=verify_lock,
            check_ab_checkpoints=check_ab_checkpoints,
        ),
    )


def _verify_acceptance_impl(directory, root=ROOT, lock_name=DEFAULT_LOCK_NAME, reaudit=True, overrides=None):
    """Verify explicit formal acceptance: contained report/audit files, locked runtime, passing audit, and report SHA-256."""
    require_debug_assertions()
    validate_lock_name(lock_name)
    root_orig = root
    root = Path(root_orig).resolve(strict=True)
    out_dir = _validate_out_dir_chain(Path(directory), root, root_orig=root_orig)

    verify_lock = True if overrides is None else overrides.verify_lock
    if verify_lock:
        verify_lock_bundle(root=root, lock_name=lock_name)
    lock_dir = root / "results" / lock_name
    checked_directory(lock_dir)
    lock = decode_json((lock_dir / "lock.json").read_text(encoding="utf-8"))
    require(isinstance(lock, dict), "lock.json must be an object")

    live_env = collect_runtime_environment()
    require(type(live_env["threads"]) is int and live_env["threads"] == lock["budget"]["threads"]
            and live_env["deterministic_algorithms"] is lock["budget"]["deterministic_algorithms"]
            and live_env["torch"] == lock["environment"]["torch"],
            "Formal acceptance verification requires locked runtime torch, threads, and determinism")

    require((out_dir / "report.json").exists(), "Formal acceptance rejected: missing report.json")
    require((out_dir / "audit.json").exists(), "Formal acceptance rejected: missing audit.json")
    report_path = _validate_contained_file(out_dir, root, _norm_rel(str(out_dir.relative_to(root) / "report.json")), "report.json")
    audit_path = _validate_contained_file(out_dir, root, _norm_rel(str(out_dir.relative_to(root) / "audit.json")), "audit.json")

    report = decode_json(report_path.read_text(encoding="utf-8"))
    audit_doc = decode_json(audit_path.read_text(encoding="utf-8"))
    require(isinstance(report, dict) and isinstance(audit_doc, dict), "report.json and audit.json must be objects")
    require(report.get("status") == "completed_unaudited",
            f"Formal acceptance rejected: invalid report status {report.get('status')!r}")
    require(audit_doc.get("passed") is True
            and audit_doc.get("formal_acceptance") is True
            and audit_doc.get("audit_scope") == "formal"
            and audit_doc.get("runtime_matches_locked_torch") is True
            and audit_doc.get("frozen_ffn_bitwise_unchanged") is True
            and audit_doc.get("paired_streams_exact") is True
            and audit_doc.get("decision_reconciled") is True,
            "Formal acceptance rejected: audit.json is not a passing formal audit")
    audit_env = audit_doc.get("audit_environment")
    require(isinstance(audit_env, dict) and audit_env.get("torch") == lock["environment"]["torch"],
            "Formal acceptance rejected: audit_environment torch mismatch")
    actual_report_sha = digest(report_path)
    require(audit_doc.get("report_sha256") == actual_report_sha,
            "Formal acceptance rejected: audit.json report_sha256 does not match report.json")

    if reaudit:
        _audit_directory_impl(
            out_dir,
            root=root,
            lock_name=lock_name,
            save=False,
            diagnostic_cross_runtime=False,
            overrides=overrides,
        )

    return dict(
        accepted=True,
        output_directory=str(out_dir.relative_to(root).as_posix()),
        report_sha256=actual_report_sha,
        audit_sha256=digest(audit_path),
        formal_updates=report.get("formal_updates"),
        decision=report.get("decision"),
        audit=audit_doc,
    )


def verify_acceptance(directory, root=ROOT, lock_name=DEFAULT_LOCK_NAME, reaudit=True):
    """Verify formal acceptance for a completed C/D output directory against the verified lock."""
    return _verify_acceptance_impl(directory, root=root, lock_name=lock_name, reaudit=reaudit, overrides=None)


def _verify_acceptance_for_tests(
    directory,
    root=ROOT,
    lock_name=DEFAULT_LOCK_NAME,
    ab_report_rel=f"{CONTINUATION_REL}/report.json",
    expected_counts=COUNTS,
    expected_stages=STAGES,
    expected_stage_updates=STAGE_UPDATES,
    expected_stage_examples=STAGE_EXAMPLES,
    expected_stage_targets=STAGE_TARGETS,
    verify_lock=True,
    check_ab_checkpoints=True,
    reaudit=True,
):
    return _verify_acceptance_impl(
        directory,
        root=root,
        lock_name=lock_name,
        reaudit=reaudit,
        overrides=_AuditOverrides(
            ab_report_rel=ab_report_rel,
            expected_counts=dict(expected_counts),
            expected_stages=tuple(expected_stages),
            expected_stage_updates=expected_stage_updates,
            expected_stage_examples=expected_stage_examples,
            expected_stage_targets=expected_stage_targets,
            verify_lock=verify_lock,
            check_ab_checkpoints=check_ab_checkpoints,
        ),
    )


def ab_report_updates(ab_report):
    return sum(r["updates"] for r in ab_report["runs"])


def main(argv=None):
    require_debug_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Exclusive C/D result directory to audit")
    parser.add_argument("--lock-name", default=DEFAULT_LOCK_NAME)
    parser.add_argument("--no-save", action="store_true", help="Do not write audit.json")
    parser.add_argument(
        "--diagnostic-cross-runtime",
        action="store_true",
        help="Run a non-accepting diagnostic audit across PyTorch builds (requires --no-save)",
    )
    parser.add_argument(
        "--verify-accepted",
        action="store_true",
        help="Verify formal acceptance of an already-audited directory (requires valid audit.json)",
    )
    args = parser.parse_args(argv)
    if args.verify_accepted and (args.no_save or args.diagnostic_cross_runtime):
        parser.error("--verify-accepted cannot be combined with --no-save or --diagnostic-cross-runtime")
    if args.diagnostic_cross_runtime and not args.no_save:
        parser.error("--diagnostic-cross-runtime requires --no-save")
    validate_lock_name(args.lock_name)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.verify_accepted:
        result = verify_acceptance(args.directory, lock_name=args.lock_name, reaudit=True)
    else:
        result = audit_directory(
            args.directory,
            lock_name=args.lock_name,
            save=not args.no_save,
            diagnostic_cross_runtime=args.diagnostic_cross_runtime,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
