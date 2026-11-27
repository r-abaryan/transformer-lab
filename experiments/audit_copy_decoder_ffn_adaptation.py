"""Independent saved-checkpoint, bitwise FFN-freeze, stream, and quartet-gate auditor."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("experiments", "tests"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from audit_copy_position_exposure import close, own_measures
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
from safe_runner import checked_directory
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


def _number(value, context):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite or nonnumeric value: {context}")
    return value


def _close_scalar(a, b, context, tol=TOLERANCE):
    require(abs(_number(a, context) - _number(b, context)) <= tol, f"Numeric mismatch: {context}")


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


def audit_supplied_cell(model, rows, arm, stored, expected, pos=None):
    """Non-strippable wrapper exercising reference evaluator, own_measures, and independent audit_cell."""
    require_debug_assertions()
    require(isinstance(stored, dict) and {"logits", "scores", "generated"}.issubset(stored),
            "Stored score payload missing required keys")
    err, metric_err = audit_cell(model, rows, pos, arm, stored, expected)
    require(isinstance(err, float) and math.isfinite(err) and err <= 1e-9,
            f"Reference logit/score error out of tolerance: {err}")
    require(isinstance(metric_err, float) and math.isfinite(metric_err) and metric_err <= 1e-9,
            f"Reference metric error out of tolerance: {metric_err}")
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


def capture_ffn_snapshot(checkpoint):
    """Capture frozen FFN parameter tensors and Adam state directly by parameter ID."""
    probe = make_model("rope")
    named = tuple(probe.named_parameters())
    param_count = len(named)
    groups = checkpoint["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["params"] == list(range(param_count)),
            "Unexpected optimizer param group contract")
    require(groups[0]["lr"] == 0.001, "Unexpected Adam lr in checkpoint")
    opt_state = checkpoint["optimizer"]["state"]
    require(set(opt_state.keys()) == set(range(param_count)), "Incomplete optimizer state table")
    ffn_ids = {name: idx for idx, (name, _) in enumerate(named) if name in FFN_NAMES}
    require(set(ffn_ids) == set(FFN_NAMES), "Missing FFN parameter name in model")
    return {
        name: dict(
            param_id=pid,
            tensor=checkpoint["model"][name].detach().clone(),
            adam=dict(opt_state[pid]),
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
    groups = cp["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["params"] == list(range(param_count)),
            "Optimizer parameter ordering violated")
    require(groups[0]["lr"] == 0.001, "Optimizer lr drifted")
    opt_state = cp["optimizer"]["state"]
    require(set(opt_state.keys()) == set(range(param_count)), "Optimizer state keys mismatch")

    trainable_moved = False
    for pid, (name, _) in enumerate(named):
        state = opt_state[pid]
        if name in FFN_NAMES:
            snap = start_ffn[name]
            require(same(cp["model"][name], snap["tensor"]),
                    f"Frozen decoder FFN parameter moved: {name}")
            require(same(state, snap["adam"]),
                    f"Frozen decoder FFN Adam state moved: {name}")
            require(int(state["step"]) == expected_start_step,
                    f"Frozen decoder FFN Adam step advanced: {name}")
        else:
            require(int(state["step"]) == expected_step,
                    f"Trainable parameter Adam step mismatch: {name}")
            if not same(cp["model"][name], prior_cp["model"][name]):
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


def audit_csv(csv_text, cd_runs, expected_counts=COUNTS):
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
    all_checkpoints = [6000, *stages]
    ab_by_key = {(r["arm"], r["step"]): r["evaluation"]["metrics"] for r in ab_report["runs"]}
    cd_by_key = {(r["cell"], r["step"]): r["evaluation"]["metrics"] for r in cd_report["runs"]}

    short_failures = []
    worst_short = 1.0
    for step in all_checkpoints:
        step_metrics = (
            {c: ab_report["initial"]["metrics"] for c in ("A", "B", "C", "D")}
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
                        short_failures.append(dict(cell=cell, step=step, pool=pool, group=group, exact_accuracy=acc))

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
        repair[cell] = all(
            ms[pool][grp][metric] >= threshold
            for pool, grp in (("test_8", "distinct"), ("test_8", "repeated"), ("test_10", "repeated"))
            for metric in ("exact_accuracy", "on_time_eos")
        )

    return dict(
        decision_step=final_step,
        primary=dict(
            accuracy=acc10,
            b_minus_a=ab_effect,
            d_minus_c=dc_effect,
            interaction=interaction,
            material_attenuation_min_gain=gates_spec["material_attenuation_min_gain"],
            material_attenuation=(
                interaction >= gates_spec["material_attenuation_min_gain"] and ab_effect > 0.0
            ),
            positive_exposure_benefit=ab_effect > 0.0,
            paired=paired10,
        ),
        short_preservation=dict(
            min_accuracy=gates_spec["short_preservation_min_accuracy"],
            worst_accuracy=worst_short,
            passed=not short_failures,
            failures=short_failures,
        ),
        secondary_length8=len8_groups,
        historical_repair_passed=repair,
    )


def audit_directory(
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
):
    """Independently verify checkpoints, frozen FFN state, streams, evaluations, CSV, and gates."""
    require_debug_assertions()
    root = Path(root).resolve(strict=True)
    out_dir = Path(directory)
    checked_directory(out_dir)
    if save:
        require(not (out_dir / "audit.json").exists(), "Refusing to overwrite existing audit.json")

    if verify_lock:
        verify_lock_bundle(root=root, lock_name=lock_name)
    lock_dir = root / "results" / lock_name
    checked_directory(lock_dir)
    lock = decode_json((lock_dir / "lock.json").read_text(encoding="utf-8"))
    require(isinstance(lock, dict), "lock.json must be an object")

    plan = decode_json((out_dir / "plan.json").read_text(encoding="utf-8"))
    report = decode_json((out_dir / "report.json").read_text(encoding="utf-8"))
    csv_raw = (out_dir / "metrics.csv").read_bytes()
    require(isinstance(plan, dict) and isinstance(report, dict), "plan.json and report.json must be objects")
    require(report.get("status") == "complete", f"Incomplete report status: {report.get('status')}")
    require(plan.get("lock_sha256") == digest(lock_dir / "lock.json"), "Plan lock_sha256 mismatch")
    require(report.get("plan_sha256") == digest(out_dir / "plan.json"), "Report plan_sha256 mismatch")
    require(plan.get("preflight_sha256") == digest(out_dir / "preflight.json"), "Plan preflight_sha256 mismatch")
    require(report.get("metrics_sha256") == hashlib.sha256(csv_raw).hexdigest(), "Report metrics_sha256 mismatch")

    require(plan.get("protected_hashes") == lock["protected_hashes"], "Protected hash manifest drifted from lock")
    verify_protected_hashes(root, plan["protected_hashes"])

    data_raw = json.loads((root / DATA_REL).read_text(encoding="utf-8"))
    data_pools = {k: v for k, v in data_raw["pools"].items() if not k.startswith("uniform_")}
    require(digest(root / DATA_REL) == lock["historical_artifact_hashes"][DATA_REL], "Data SHA-256 mismatch")
    train_pools = {k: torch.tensor(v) for k, v in data_pools.items() if k.startswith("train_")}

    start_path = root / START_REL
    start_sha = digest(start_path)
    require(start_sha == lock["historical_artifact_hashes"][START_REL], "Start checkpoint SHA-256 mismatch")
    start_cp = torch.load(start_path, weights_only=True)
    require(start_cp.get("step") == 6000, "Start checkpoint step must be 6000")
    start_ffn, _ = capture_ffn_snapshot(start_cp)

    ab_report_path = root / _norm_rel(ab_report_rel)
    require(digest(ab_report_path) == lock["historical_artifact_hashes"][_norm_rel(ab_report_rel)],
            "Historical A/B report SHA-256 mismatch")
    ab_report = json.loads(ab_report_path.read_text(encoding="utf-8"))
    require(ab_report.get("status") == "complete", "Historical A/B report is not complete")
    require(plan.get("historical_ab_report_sha256") == digest(ab_report_path), "Plan historical A/B report hash mismatch")
    require(digest(root / _norm_rel(ab_report["initial"]["scores"])) == ab_report["initial"]["scores_sha256"],
            "Historical A/B initial scores SHA-256 mismatch")
    for ab_run in ab_report["runs"]:
        ab_scores_rel = _norm_rel(ab_run["evaluation"]["scores"])
        require(digest(root / ab_scores_rel) == ab_run["evaluation"]["scores_sha256"],
                f"Historical A/B scores SHA-256 mismatch: {ab_scores_rel}")

    # Verify initial update-6000 evaluation matches historical initial evaluation.
    start_model = make_model("rope").eval()
    start_model.load_state_dict(start_cp["model"], strict=True)
    init_ev = report["initial"]
    require(digest(root / _norm_rel(init_ev["scores"])) == init_ev["scores_sha256"], "Initial scores SHA-256 mismatch")
    validate_metrics_mapping(init_ev["metrics"], expected_counts=expected_counts)
    init_saved = torch.load(root / _norm_rel(init_ev["scores"]), weights_only=True)
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
                and init_saved[pool_name]["generated"] == hist_init_scores[pool_name]["generated"],
                f"Initial evaluation diverges from historical continuation at {pool_name}")

    runs = report.get("runs")
    expected_runs = {(cell, arm, stop) for cell, arm in CD_CELLS for stop in expected_stages}
    require(isinstance(runs, list) and len(runs) == len(expected_runs), "Unexpected number of C/D stage runs")
    runs_by_key = {}
    for run in runs:
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
            ckpt_path = root / _norm_rel(run["checkpoint"])
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
                ab_cp = torch.load(root / _norm_rel(ab_stage[arm]["checkpoint"]), weights_only=True)
                require(torch.equal(cp["generator"], ab_cp["generator"])
                        and torch.equal(cp["position_generator"], ab_cp["position_generator"]),
                        f"RNG state diverged from historical checkpoint at {cell}@{stop}")

            ev = run["evaluation"]
            scores_path = root / _norm_rel(ev["scores"])
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
        prev_step = stop

    csv_rows = audit_csv(csv_raw.decode("utf-8"), runs, expected_counts=expected_counts)
    own_decision = _recompute_decision(root, ab_report, report, lock["gates"], expected_stages, expected_counts, data_pools)
    close(own_decision, report["decision"])

    verify_protected_hashes(root, plan["protected_hashes"])
    result = dict(
        passed=True,
        cells_audited=cells_audited,
        csv_rows=csv_rows,
        max_reference_error=max_error,
        max_metric_error=max_metric_error,
        cd_updates=sum(r["updates"] for r in runs),
        cd_examples=sum(r["examples"] for r in runs),
        cd_targets=sum(r["active_targets"] for r in runs),
        ab_plus_cd_total_updates=ab_report_updates(ab_report) + sum(r["updates"] for r in runs),
        frozen_ffn_bitwise_unchanged=True,
        paired_streams_exact=True,
        decision_reconciled=True,
        protected_files=len(plan["protected_hashes"]),
        report_sha256=digest(out_dir / "report.json"),
        limitation="Saved-checkpoint forward/decoding/metrics/stream/frozen-FFN audit; gradient step equivalence checked by runner contract and tests.",
    )
    if save:
        write(out_dir / "audit.json", result)
    return result


def ab_report_updates(ab_report):
    return sum(r["updates"] for r in ab_report["runs"])


def main(argv=None):
    require_debug_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Exclusive C/D result directory to audit")
    parser.add_argument("--lock-name", default=DEFAULT_LOCK_NAME)
    parser.add_argument("--no-save", action="store_true", help="Do not write audit.json")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = audit_directory(args.directory, lock_name=args.lock_name, save=not args.no_save)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
