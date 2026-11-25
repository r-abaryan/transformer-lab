"""Read-only historical verification, A/B replay, and exclusive lock for decoder FFN adaptation."""
import argparse
import copy
import hashlib
import json
import os
import platform
import struct
import sys
import time
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("experiments", "tests"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from audit_copy_position_sites import audit_cell
from copy_decoder_ffn_adaptation import (
    FFN_NAMES,
    check_order,
    preflight as preflight_check,
    process_memory,
    require,
    restore,
    same,
    targets,
)
from copy_position_sites import digest, evaluate, forward, inputs, make_model, positions, sample, write
from safe_runner import checked_directory, fresh_directory, validate_target
from validate_copy_exposure_continuation import COUNTS, decode_json, validate_directory


def require_debug_assertions():
    """Fail closed when Python -O disables assert statements inside frozen historical auditors."""
    if not __debug__:
        raise RuntimeError("Python -O / optimized mode is not permitted; required audit assertions would be disabled.")

ARMS = ("cross_only", "both")
STAGES = (6500, 7000, 7500, 8000)
STAGE_UPDATES = 500
STAGE_EXAMPLES = 79750
STAGE_TARGETS = 420000
START_REL = "results/copy_learning_trajectory/3767_cross_only_6000.pt"
DATA_REL = "results/copy_position_sites/data.json"
CONTINUATION_REL = "results/copy_exposure_continuation"
TRAJECTORY_AUDIT_REL = "results/copy_learning_trajectory/persisted-audit.json"
DEFAULT_LOCK_NAME = "copy_decoder_ffn_adaptation"
STUDY_FILES = (
    "docs/protocols/copy-decoder-ffn-adaptation.md",
    "experiments/copy_decoder_ffn_adaptation.py",
    "experiments/prepare_copy_decoder_ffn_adaptation.py",
    "tests/test_copy_decoder_ffn_adaptation.py",
    "tests/test_copy_decoder_ffn_lock.py",
)


def _norm_rel(path_str):
    return path_str.replace("\\", "/")


def verify_pools(pools_payload, expected_counts=COUNTS):
    """Validate non-uniform pool keys, distinct-first/repeated-second ordering, labels, and target counts."""
    require(isinstance(pools_payload, dict), "Pools must be a mapping")
    require(set(pools_payload) == set(expected_counts), "Unexpected or missing pool keys")
    total_sequences = 0
    total_targets = 0
    for name, (n_distinct, n_repeated) in expected_counts.items():
        rows = pools_payload[name]
        length = int(name.rsplit("_", 1)[1])
        expected_len = n_distinct + n_repeated
        require(isinstance(rows, list) and len(rows) == expected_len, f"Wrong row count in {name}")
        for idx, row in enumerate(rows):
            require(
                isinstance(row, list) and len(row) == length
                and all(type(v) is int and 3 <= v <= 10 for v in row),
                f"Invalid token sequence in {name}[{idx}]",
            )
            is_distinct = len(set(row)) == length
            require(
                is_distinct == (idx < n_distinct),
                f"Pool order violation in {name}[{idx}]: distinct first half, repeated second required",
            )
        tensor_rows = torch.tensor(rows)
        prefix, labels = inputs(tensor_rows)
        require(prefix.tolist() == [[1] + r for r in rows], f"BOS prefix mismatch in {name}")
        require(labels.tolist() == [r + [2] for r in rows], f"EOS/content label mismatch in {name}")
        require(labels.numel() == expected_len * (length + 1), f"Target count mismatch in {name}")
        total_sequences += expected_len
        total_targets += labels.numel()
    return dict(pools=len(expected_counts), sequences=total_sequences, targets=total_targets)


def verify_start_checkpoint(checkpoint_path, expected_sha256=None, expected_step=6000):
    """Verify starting checkpoint hash, payload contract, Adam counters, and both restore modes."""
    checkpoint_path = Path(checkpoint_path)
    require(checkpoint_path.is_file(), f"Missing start checkpoint: {checkpoint_path}")
    actual_sha = digest(checkpoint_path)
    if expected_sha256 is not None:
        require(actual_sha == expected_sha256, "Start checkpoint SHA-256 mismatch")

    cp = torch.load(checkpoint_path, weights_only=True)
    require(
        isinstance(cp, dict)
        and set(cp.keys()) == {"model", "optimizer", "step", "generator", "position_generator"},
        "Unexpected checkpoint keys",
    )
    require(type(cp["step"]) is int and cp["step"] == expected_step, "Unexpected start step")

    probe = make_model("rope")
    model_params = tuple(probe.parameters())
    expected_param_count = len(model_params)
    groups = cp["optimizer"]["param_groups"]
    require(len(groups) == 1, "Historical Adam must have one parameter group")
    require(groups[0]["params"] == list(range(expected_param_count)), "Saved optimizer param IDs mismatch")
    require(groups[0]["lr"] == 0.001, "Saved Adam lr mismatch")
    opt_state = cp["optimizer"]["state"]
    require(set(opt_state.keys()) == set(range(expected_param_count)), "Adam state keys mismatch")
    for param_id, param in enumerate(model_params):
        state = opt_state[param_id]
        require(int(state["step"]) == expected_step, f"Adam step mismatch for param {param_id}")
        for moment_key in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_key)
            require(
                torch.is_tensor(moment) and moment.shape == param.shape and moment.dtype == param.dtype
                and bool(torch.isfinite(moment).all()),
                f"Adam {moment_key} tensor mismatch for param {param_id}",
            )

    snapshot_before = copy.deepcopy(cp)
    for frozen in (False, True):
        model, opt, generators = restore(cp, lambda: make_model("rope"), frozen=frozen)
        check_order(model, opt)
        ffn = targets(model)
        for name, param in model.named_parameters():
            if name in ffn:
                require(param.requires_grad is (not frozen) and param.grad is None, f"Wrong FFN grad flag: {name}")
            else:
                require(param.requires_grad is True, f"Extra parameter frozen: {name}")
        require(len(generators) == 2, "Expected generator and position_generator")
    require(same(cp, snapshot_before), "Checkpoint dict mutated during restore verification")
    require(digest(checkpoint_path) == actual_sha, "Start checkpoint file changed during verification")
    return dict(
        sha256=actual_sha,
        step=cp["step"],
        parameter_count=expected_param_count,
        ffn_parameter_count=len(FFN_NAMES),
        trainable_when_frozen=expected_param_count - len(FFN_NAMES),
    )


def verify_protected_hashes(root, mapping):
    """Verify every recorded path in a protected manifest matches its SHA-256 on disk."""
    checked = {}
    for raw_rel, expected_sha in mapping.items():
        rel = _norm_rel(raw_rel)
        target = root / rel
        require(target.is_file(), f"Missing protected file: {rel}")
        actual = digest(target)
        require(actual == expected_sha, f"Protected file SHA-256 mismatch: {rel}")
        checked[rel] = actual
    return checked


def verify_historical_lineage(root=ROOT):
    """Read-only verification of historical continuation artifacts, protected hashes, pools, and start checkpoint."""
    require_debug_assertions()
    root = Path(root)
    cont_dir = root / CONTINUATION_REL
    start_path = root / START_REL
    data_path = root / DATA_REL
    traj_audit_path = root / TRAJECTORY_AUDIT_REL

    for required in (cont_dir / "plan.json", cont_dir / "report.json", cont_dir / "audit.json",
                     cont_dir / "preflight.json", cont_dir / "metrics.csv",
                     start_path, data_path, traj_audit_path):
        require(required.is_file(), f"Missing historical artifact: {required}")

    continuation_validation = validate_directory(cont_dir)
    plan = json.loads((cont_dir / "plan.json").read_text(encoding="utf-8"))
    report = json.loads((cont_dir / "report.json").read_text(encoding="utf-8"))
    audit = json.loads((cont_dir / "audit.json").read_text(encoding="utf-8"))
    traj_audit = json.loads(traj_audit_path.read_text(encoding="utf-8"))

    require(digest(cont_dir / "plan.json") == report["plan_sha256"], "Continuation plan SHA-256 mismatch")
    require(digest(cont_dir / "preflight.json") == plan["preflight_sha256"], "Continuation preflight SHA-256 mismatch")
    require(digest(cont_dir / "report.json") == audit["report_sha256"], "Continuation report SHA-256 mismatch")
    require(audit.get("passed") is True, "Continuation audit did not pass")

    protected_checked = verify_protected_hashes(root, plan["protected"])
    start_sha = protected_checked[START_REL]
    traj_matches = [
        sha for raw_p, sha in traj_audit["snapshot_artifact_hashes"].items()
        if _norm_rel(raw_p) == START_REL
    ]
    require(traj_matches == [start_sha], "Start checkpoint does not match trajectory persisted audit")

    data_raw = json.loads(data_path.read_text(encoding="utf-8"))
    non_uniform = {k: v for k, v in data_raw["pools"].items() if not k.startswith("uniform_")}
    pool_summary = verify_pools(non_uniform)
    start_summary = verify_start_checkpoint(start_path, expected_sha256=start_sha, expected_step=6000)

    init_scores_rel = _norm_rel(report["initial"]["scores"])
    require(digest(root / init_scores_rel) == report["initial"]["scores_sha256"], "Initial scores SHA-256 mismatch")

    artifact_hashes = {
        START_REL: start_sha,
        DATA_REL: protected_checked[DATA_REL],
        TRAJECTORY_AUDIT_REL: digest(traj_audit_path),
        f"{CONTINUATION_REL}/plan.json": digest(cont_dir / "plan.json"),
        f"{CONTINUATION_REL}/preflight.json": plan["preflight_sha256"],
        f"{CONTINUATION_REL}/report.json": audit["report_sha256"],
        f"{CONTINUATION_REL}/metrics.csv": report["metrics_sha256"],
        f"{CONTINUATION_REL}/audit.json": digest(cont_dir / "audit.json"),
        init_scores_rel: report["initial"]["scores_sha256"],
    }
    for run in report["runs"]:
        ckpt_rel = _norm_rel(run["checkpoint"])
        scores_rel = _norm_rel(run["evaluation"]["scores"])
        require((root / ckpt_rel).is_file() and (root / scores_rel).is_file(), f"Missing run artifact: {ckpt_rel}")
        prior_rel = START_REL if run["step"] == STAGES[0] else f"{CONTINUATION_REL}/3767_{run['arm']}_{run['step'] - STAGE_UPDATES}.pt"
        require((root / prior_rel).is_file(), f"Missing prior checkpoint: {prior_rel}")
        require(run["start_sha256"] == digest(root / prior_rel), f"Segment start_sha256 mismatch for {run['arm']}@{run['step']}")
        ckpt_sha = digest(root / ckpt_rel)
        scores_sha = digest(root / scores_rel)
        require(ckpt_sha == run["checkpoint_sha256"], f"Checkpoint SHA-256 mismatch: {ckpt_rel}")
        require(scores_sha == run["evaluation"]["scores_sha256"], f"Scores SHA-256 mismatch: {scores_rel}")
        artifact_hashes[ckpt_rel] = ckpt_sha
        artifact_hashes[scores_rel] = scores_sha

    return dict(
        passed=True,
        start_checkpoint=start_summary,
        pools=pool_summary,
        continuation_validation=continuation_validation,
        protected_count=len(protected_checked),
        protected_hashes=protected_checked,
        artifact_hashes=artifact_hashes,
    )


def replay_segment(model, opt, gen, pg, train_pools, begin_step, stop_step, arm):
    """Replay one continuation segment while independently checking sampler draws, RNG consumption, and CE updates."""
    require(arm in ARMS, f"Unknown arm: {arm}")
    require(stop_step > begin_step, "stop_step must exceed begin_step")
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

        require(n == expected_n, f"Step {step} cycle length mismatch")
        require(torch.equal(idx, expected_idx), f"Step {step} sampled indices mismatch")
        require(packed == struct.pack("210d", *ranks.flatten().tolist()), f"Step {step} packed ranks mismatch")
        require(torch.equal(pos, expected_pos), f"Step {step} position sample mismatch")
        require(
            torch.equal(gen.get_state(), rg.get_state()) and torch.equal(pg.get_state(), rpg.get_state()),
            f"Step {step} generator state mismatch",
        )

        stream.update(json.dumps([n, idx.tolist()]).encode())
        pstream.update(json.dumps(pos.tolist()).encode())

        x = train_pools[f"train_{n}"][idx]
        prefix, y = inputs(x)
        opt.zero_grad(set_to_none=True)
        z = forward(model, x, prefix, pos if arm == "both" else None, pos)[0]
        loss = F.cross_entropy(z.flatten(0, 1), y.flatten())
        require(bool(torch.isfinite(loss)), f"Nonfinite loss at step {step}")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                f"Missing/nonfinite gradient at step {step}")
        opt.step()
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


def _max_tensor_diff(a, b):
    if torch.is_tensor(a):
        return float((a.detach() - b.detach()).abs().max()) if a.dtype.is_floating_point else int((a != b).sum())
    if isinstance(a, dict):
        return max((_max_tensor_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        return max((_max_tensor_diff(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def _audit_saved_eval(model, eval_pools, saved_scores, expected_metrics, context_label):
    """Verify evaluated metrics and saved score tensors with non-strippable checks plus independent audit_cell."""
    require_debug_assertions()
    cells = 0
    max_error = 0.0
    model.eval()
    for pool_name, rows in eval_pools.items():
        require(pool_name in saved_scores and pool_name in expected_metrics,
                f"Missing saved score pool {pool_name} in {context_label}")
        metrics, actual_saved = evaluate(model, rows, "cross_only")
        require(metrics == expected_metrics[pool_name], f"Eval metrics mismatch for {context_label}/{pool_name}")
        stored = saved_scores[pool_name]
        require(actual_saved["generated"] == stored["generated"],
                f"Saved generated sequences mismatch for {context_label}/{pool_name}")
        require(torch.equal(actual_saved["logits"].argmax(-1), stored["logits"].argmax(-1)),
                f"Saved argmax mismatch for {context_label}/{pool_name}")
        logit_diff = float((actual_saved["logits"] - stored["logits"]).abs().max())
        score_diff = float((actual_saved["scores"] - stored["scores"]).abs().max())
        require(logit_diff <= 1e-9 and score_diff <= 1e-9,
                f"Saved logit/score tensor mismatch for {context_label}/{pool_name}")
        err, metric_err = audit_cell(model, rows, None, "cross_only", stored, metrics)
        require(isinstance(err, float) and err <= 1e-9 and isinstance(metric_err, float) and metric_err <= 1e-9,
                f"Reference audit tolerance exceeded for {context_label}/{pool_name}")
        max_error = max(max_error, err, logit_diff, score_diff)
        cells += 1
    return cells, max_error


def replay_ab(root=ROOT, verify_eval=True):
    """Verify exact A/B historical continuation replay (steps 6001-8000) without mutating frozen artifacts."""
    require_debug_assertions()
    tick = time.perf_counter()
    memory_before = process_memory()
    root = Path(root)
    lineage = verify_historical_lineage(root)

    data_raw = json.loads((root / DATA_REL).read_text(encoding="utf-8"))
    eval_pools = {k: v for k, v in data_raw["pools"].items() if not k.startswith("uniform_")}
    train_pools = {k: torch.tensor(v) for k, v in eval_pools.items() if k.startswith("train_")}
    report = json.loads((root / CONTINUATION_REL / "report.json").read_text(encoding="utf-8"))
    start_cp = torch.load(root / START_REL, weights_only=True)

    eval_cells = 0
    max_eval_ref_error = 0.0
    if verify_eval:
        init_model, _, _ = restore(start_cp, lambda: make_model("rope"), frozen=False)
        init_scores = torch.load(root / _norm_rel(report["initial"]["scores"]), weights_only=True)
        cells, err = _audit_saved_eval(
            init_model, eval_pools, init_scores, report["initial"]["metrics"], "initial@6000"
        )
        eval_cells += cells
        max_eval_ref_error = max(max_eval_ref_error, err)

    stage_records = []
    arm_states = {}
    prior_sha = {}
    for arm in ARMS:
        model, opt, (gen, pg) = restore(start_cp, lambda: make_model("rope"), frozen=False)
        arm_states[arm] = dict(model=model, opt=opt, gen=gen, pg=pg)
        prior_sha[arm] = lineage["start_checkpoint"]["sha256"]

    prev_step = 6000
    for stop in STAGES:
        stage_pair = []
        for arm in ARMS:
            state = arm_states[arm]
            seg_tick = time.perf_counter()
            stream_hex, pstream_hex, examples, targets_count, replayed_cp = replay_segment(
                state["model"], state["opt"], state["gen"], state["pg"],
                train_pools, prev_step, stop, arm,
            )
            seg_seconds = time.perf_counter() - seg_tick
            expected_run = next(r for r in report["runs"] if r["arm"] == arm and r["step"] == stop)
            require(expected_run["start_sha256"] == prior_sha[arm], f"Segment start_sha256 mismatch for {arm}@{stop}")
            require(stop - prev_step == expected_run["updates"] == STAGE_UPDATES, "Segment updates mismatch")
            require(examples == expected_run["examples"] == STAGE_EXAMPLES, "Segment examples mismatch")
            require(targets_count == expected_run["active_targets"] == STAGE_TARGETS, "Segment targets mismatch")
            require(stream_hex == expected_run["stream"], f"Stream hash mismatch for {arm}@{stop}")
            require(pstream_hex == expected_run["position_stream"], f"Position stream hash mismatch for {arm}@{stop}")

            saved_cp = torch.load(root / _norm_rel(expected_run["checkpoint"]), weights_only=True)
            require(same(replayed_cp, saved_cp), f"Bitwise checkpoint mismatch for {arm}@{stop}")
            prior_sha[arm] = expected_run["checkpoint_sha256"]

            if verify_eval:
                saved_scores = torch.load(root / _norm_rel(expected_run["evaluation"]["scores"]), weights_only=True)
                cells, err = _audit_saved_eval(
                    state["model"], eval_pools, saved_scores, expected_run["evaluation"]["metrics"], f"{arm}@{stop}"
                )
                eval_cells += cells
                max_eval_ref_error = max(max_eval_ref_error, err)

            record = dict(
                arm=arm,
                begin=prev_step,
                step=stop,
                updates=STAGE_UPDATES,
                examples=examples,
                targets=targets_count,
                stream=stream_hex,
                position_stream=pstream_hex,
                bitwise_exact=True,
                max_abs_diff=_max_tensor_diff(replayed_cp["model"], saved_cp["model"]),
                seconds=seg_seconds,
            )
            stage_pair.append(record)
            stage_records.append(record)
        require(stage_pair[0]["stream"] == stage_pair[1]["stream"], f"Paired stream mismatch at {stop}")
        require(stage_pair[0]["position_stream"] == stage_pair[1]["position_stream"], f"Paired pstream mismatch at {stop}")
        prev_step = stop

    # Re-verify every protected file remained untouched after replay.
    verify_protected_hashes(root, lineage["protected_hashes"])
    memory_after = process_memory()
    if sys.platform == "win32":
        require(memory_before["available"] and memory_after["available"]
                and memory_after["peak_working_set_bytes"] > 0, "Windows memory measurement failed")

    per_arm_updates = len(STAGES) * STAGE_UPDATES
    return dict(
        passed=True,
        formal_ready=False,
        formal_updates=0,
        replayed_arms=list(ARMS),
        stages=list(STAGES),
        replayed_updates_per_arm=per_arm_updates,
        replayed_updates_total=per_arm_updates * len(ARMS),
        bitwise_exact_all=True,
        eval_cells_checked=eval_cells,
        max_eval_reference_error=max_eval_ref_error,
        stage_records=stage_records,
        lineage_protected_files=lineage["protected_count"],
        artifact_hashes=lineage["artifact_hashes"],
        protected_hashes=lineage["protected_hashes"],
        memory=dict(before=memory_before, after=memory_after),
        seconds=time.perf_counter() - tick,
        environment=dict(
            python=sys.version,
            torch=torch.__version__,
            platform=platform.platform(),
            cpu=platform.processor(),
            threads=torch.get_num_threads(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            dtype="float64",
            device="cpu",
        ),
        limitation="Read-only historical A/B replay verification only; no C/D training or formal execution authorization.",
    )


def build_lock_payload(replay_report, preflight_report, root=ROOT, lock_dir_rel=f"results/{DEFAULT_LOCK_NAME}"):
    """Build the frozen lock specification prior to separate formal execution authorization."""
    study_hashes = {}
    for rel in STUDY_FILES:
        target = root / rel
        require(target.is_file(), f"Missing study file: {rel}")
        study_hashes[rel] = digest(target)

    return dict(
        schema_version=1,
        status="locked_awaiting_authorization",
        formal_ready=False,
        formal_authorized=False,
        formal_updates=0,
        output_directory=lock_dir_rel,
        start_checkpoint=START_REL,
        data_file=DATA_REL,
        ab_replay_source=CONTINUATION_REL,
        ab_replay_passed=replay_report["passed"],
        ab_replay_bitwise_exact=replay_report["bitwise_exact_all"],
        ab_replay_eval_cells=replay_report["eval_cells_checked"],
        preflight_passed=preflight_report["passed"],
        preflight_fixture_updates=preflight_report["fixture_updates"],
        environment=replay_report["environment"],
        budget=dict(
            cells=dict(
                A=dict(self_site_exposure=False, decoder_ffn_frozen=False),
                B=dict(self_site_exposure=True, decoder_ffn_frozen=False),
                C=dict(self_site_exposure=False, decoder_ffn_frozen=True),
                D=dict(self_site_exposure=True, decoder_ffn_frozen=True),
            ),
            frozen_parameters=list(FFN_NAMES),
            window=[6001, 8000],
            stages=list(STAGES),
            observed_checkpoints=[6000, *STAGES],
            decision_step=8000,
            optimizer="Adam",
            lr=0.001,
            dtype="float64",
            device="cpu",
            threads=1,
            deterministic_algorithms=True,
            cycle_lengths=[3, 4, 5, 6],
            batches_by_length={"3": 210, "4": 168, "5": 140, "6": 120},
            targets_per_update=840,
            updates_per_cell=2000,
            examples_per_cell=319000,
            targets_per_cell=1680000,
            quartet_max_updates=8000,
            quartet_max_examples=1276000,
            quartet_max_targets=6720000,
            ab_counts_against_quartet_max=True,
        ),
        gates=dict(
            primary_metric="test_10.all.exact_accuracy",
            primary_contrast="(B - A) - (D - C)",
            test10_sequences=128,
            material_attenuation_min_gain=13 / 128,
            require_positive_exposure_benefit="B - A > 0",
            short_preservation_min_accuracy=0.99,
            short_preservation_pools=[f"{split}_{n}" for split in ("train", "validation", "test") for n in range(3, 7)],
            short_preservation_groups=["distinct", "repeated"],
            short_preservation_every_observed_checkpoint=True,
            secondary_length8_groups=["distinct", "repeated"],
            historical_repair_threshold=0.95,
        ),
        study_hashes=study_hashes,
        historical_artifact_hashes=replay_report["artifact_hashes"],
        protected_hashes=replay_report["protected_hashes"],
        limitation="Preflight and read-only historical A/B replay locked; requires separate formal runner and explicit execution authorization before running C/D.",
    )


def create_lock(results_root=None, lock_name=DEFAULT_LOCK_NAME, root=ROOT, verify_eval=True,
                replay_fn=None, preflight_fn=None):
    """Verify historical lineage, exact A/B replay, and preflight, then atomically publish lock artifacts."""
    require_debug_assertions()
    require(verify_eval is True, "create_lock requires saved-score evaluation audit")
    base = Path(root).resolve(strict=True) / "results" if results_root is None else Path(results_root)
    root, _, target = validate_target(base, lock_name, root=root)
    lock_dir_rel = target.relative_to(root).as_posix()

    replay_report = (replay_ab(root, verify_eval=True) if replay_fn is None else replay_fn(root))
    require(replay_report.get("passed") is True and replay_report.get("bitwise_exact_all") is True,
            "Exact A/B replay must pass before locking")
    if replay_fn is None:
        require(replay_report.get("eval_cells_checked", 0) > 0,
                "Exact A/B replay must audit saved evaluation cells before locking")
    preflight_report = preflight_check() if preflight_fn is None else preflight_fn()
    require(preflight_report.get("passed") is True and preflight_report.get("formal_updates") == 0,
            "Initialized preflight must pass with zero formal updates")

    lock_payload = build_lock_payload(replay_report, preflight_report, root=root, lock_dir_rel=lock_dir_rel)

    staging_name = f"tmp_staging_{uuid.uuid4().hex[:12]}"
    staging_dir = fresh_directory(base, staging_name, root=root)
    replay_path = staging_dir / "ab_replay_verification.json"
    preflight_path = staging_dir / "preflight.json"
    tmp_lock = staging_dir / ".lock.json.tmp"
    staged_lock = staging_dir / "lock.json"
    published = False
    try:
        write(replay_path, replay_report)
        write(preflight_path, preflight_report)
        lock_payload["verification_artifacts"] = {
            f"{lock_dir_rel}/ab_replay_verification.json": digest(replay_path),
            f"{lock_dir_rel}/preflight.json": digest(preflight_path),
        }
        # Re-verify protected and study hashes after writing evidence and before publishing lock.json.
        verify_protected_hashes(root, replay_report["protected_hashes"])
        verify_protected_hashes(root, lock_payload["study_hashes"])
        write(tmp_lock, lock_payload)
        os.replace(tmp_lock, staged_lock)
        validate_target(base, lock_name, root=root)
        staging_dir.rename(target)
        published = True
    finally:
        if not published and staging_dir.exists():
            for name in (".lock.json.tmp", "lock.json", "ab_replay_verification.json", "preflight.json"):
                candidate = staging_dir / name
                if candidate.is_file():
                    candidate.unlink()
            staging_dir.rmdir()

    final_replay = target / "ab_replay_verification.json"
    final_preflight = target / "preflight.json"
    final_lock = target / "lock.json"
    return dict(
        passed=True,
        formal_ready=False,
        formal_authorized=False,
        formal_updates=0,
        lock_directory=str(target.relative_to(root).as_posix()),
        lock_sha256=digest(final_lock),
        ab_replay_sha256=digest(final_replay),
        preflight_sha256=digest(final_preflight),
    )


def verify_lock_bundle(results_root=None, lock_name=DEFAULT_LOCK_NAME, root=ROOT):
    """Read-only verification of committed lock.json, evidence artifacts, and study source hashes."""
    require_debug_assertions()
    root = Path(root).resolve(strict=True)
    base = root / "results" if results_root is None else Path(results_root)
    checked_directory(root)
    checked_directory(base)
    lock_dir = base / lock_name
    checked_directory(lock_dir)
    lock_dir_rel = lock_dir.relative_to(root).as_posix()

    expected_names = ["ab_replay_verification.json", "lock.json", "preflight.json"]
    actual_entries = sorted(p.name for p in lock_dir.iterdir())
    require(actual_entries == expected_names, f"Unexpected lock directory contents: {actual_entries}")
    for name in expected_names:
        require((lock_dir / name).is_file(), f"Lock artifact must be a regular file: {name}")

    lock = decode_json((lock_dir / "lock.json").read_text(encoding="utf-8"))
    ab = decode_json((lock_dir / "ab_replay_verification.json").read_text(encoding="utf-8"))
    pre = decode_json((lock_dir / "preflight.json").read_text(encoding="utf-8"))
    require(all(isinstance(x, dict) for x in (lock, ab, pre)), "Lock bundle files must be JSON objects")

    require(lock.get("schema_version") == 1, "Unsupported lock schema_version")
    require(lock.get("status") == "locked_awaiting_authorization", "Unexpected lock status")
    require(lock.get("formal_ready") is False and lock.get("formal_authorized") is False
            and lock.get("formal_updates") == 0, "Lock must enforce zero formal updates and false authorization")
    require(lock.get("output_directory") == lock_dir_rel, "Lock output_directory mismatch")
    require(lock.get("ab_replay_passed") is True and lock.get("ab_replay_bitwise_exact") is True
            and lock.get("ab_replay_eval_cells") == 126, "Lock A/B replay metadata mismatch")
    require(lock.get("preflight_passed") is True and lock.get("preflight_fixture_updates") == 17,
            "Lock preflight metadata mismatch")

    expected_artifacts = {
        f"{lock_dir_rel}/ab_replay_verification.json",
        f"{lock_dir_rel}/preflight.json",
    }
    require(isinstance(lock.get("verification_artifacts"), dict)
            and set(lock["verification_artifacts"]) == expected_artifacts,
            "Unexpected verification_artifacts keys in lock.json")
    verify_protected_hashes(root, lock["verification_artifacts"])

    require(isinstance(lock.get("study_hashes"), dict) and set(lock["study_hashes"]) == set(STUDY_FILES),
            "Unexpected study_hashes keys in lock.json")
    verify_protected_hashes(root, lock["study_hashes"])

    require(ab.get("passed") is True and ab.get("formal_ready") is False and ab.get("formal_updates") == 0,
            "Invalid A/B replay flags")
    require(ab.get("replayed_arms") == list(ARMS) and ab.get("stages") == list(STAGES),
            "A/B replay arms or stages mismatch")
    require(ab.get("replayed_updates_per_arm") == len(STAGES) * STAGE_UPDATES
            and ab.get("replayed_updates_total") == len(ARMS) * len(STAGES) * STAGE_UPDATES,
            "A/B replay update counts mismatch")
    require(ab.get("bitwise_exact_all") is True and ab.get("eval_cells_checked") == 126,
            "A/B replay exactness or evaluated cell count mismatch")

    stage_records = ab.get("stage_records")
    require(isinstance(stage_records, list) and len(stage_records) == len(ARMS) * len(STAGES),
            "Expected eight A/B stage records")
    seen_pairs = {}
    for rec in stage_records:
        require(isinstance(rec, dict), "Stage record must be an object")
        pair = (rec.get("arm"), rec.get("step"))
        require(pair[0] in ARMS and pair[1] in STAGES and pair not in seen_pairs,
                f"Invalid or duplicate stage record: {pair}")
        require(rec.get("updates") == STAGE_UPDATES and rec.get("examples") == STAGE_EXAMPLES
                and rec.get("targets") == STAGE_TARGETS, f"Stage record counts mismatch: {pair}")
        require(rec.get("bitwise_exact") is True and rec.get("max_abs_diff") == 0.0,
                f"Stage record not bitwise exact: {pair}")
        seen_pairs[pair] = rec
    for stop in STAGES:
        require(seen_pairs[("cross_only", stop)]["stream"] == seen_pairs[("both", stop)]["stream"]
                and seen_pairs[("cross_only", stop)]["position_stream"] == seen_pairs[("both", stop)]["position_stream"],
                f"Paired stream mismatch in lock bundle at {stop}")

    require(isinstance(lock.get("protected_hashes"), dict) and len(lock["protected_hashes"]) > 0,
            "Missing protected_hashes in lock.json")
    require(ab.get("artifact_hashes") == lock.get("historical_artifact_hashes")
            and ab.get("protected_hashes") == lock.get("protected_hashes")
            and ab.get("lineage_protected_files") == len(lock["protected_hashes"]),
            "A/B replay manifest hashes differ from lock.json")

    require(pre.get("passed") is True and pre.get("formal_ready") is False
            and pre.get("formal_updates") == 0 and pre.get("fixture_updates") == 17,
            "Invalid preflight flags in lock bundle")
    require(isinstance(pre.get("profiles"), list) and len(pre["profiles"]) == 16
            and pre.get("audit_cells") == 10, "Invalid preflight profile/audit counts")
    require(isinstance(pre.get("checks"), dict) and bool(pre["checks"])
            and all(v is True for v in pre["checks"].values()), "Incomplete preflight checks")

    return dict(
        passed=True,
        formal_ready=False,
        formal_authorized=False,
        formal_updates=0,
        lock_directory=lock_dir_rel,
        ab_replay_eval_cells=ab["eval_cells_checked"],
        stage_records=len(stage_records),
        protected_files=len(lock["protected_hashes"]),
        study_files=len(STUDY_FILES),
        lock_sha256=digest(lock_dir / "lock.json"),
        ab_replay_sha256=digest(lock_dir / "ab_replay_verification.json"),
        preflight_sha256=digest(lock_dir / "preflight.json"),
    )


def main(argv=None):
    require_debug_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify-history", "replay-ab", "lock", "verify-lock"))
    parser.add_argument("--no-eval", action="store_true", help="Skip saved-score eval audit during replay-ab")
    parser.add_argument("--save-run-name", default=None,
                        help="Optional fresh folder under results/test_runs/ to write replay-ab output")
    parser.add_argument("--lock-name", default=DEFAULT_LOCK_NAME,
                        help="Exclusive directory name under results/ when running lock or verify-lock")
    args = parser.parse_args(argv)
    if args.no_eval and args.action != "replay-ab":
        parser.error("--no-eval is only supported with replay-ab, never lock")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    if args.action == "verify-history":
        result = verify_historical_lineage(ROOT)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    if args.action == "replay-ab":
        result = replay_ab(ROOT, verify_eval=not args.no_eval)
        if args.save_run_name is not None:
            dest_dir = fresh_directory(ROOT / "results/test_runs", args.save_run_name, root=ROOT)
            write(dest_dir / "ab_replay_verification.json", result)
            result = dict(result, saved_report=str((dest_dir / "ab_replay_verification.json").relative_to(ROOT).as_posix()))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    if args.action == "verify-lock":
        result = verify_lock_bundle(lock_name=args.lock_name, root=ROOT)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0

    result = create_lock(lock_name=args.lock_name, root=ROOT)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
