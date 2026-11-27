"""Tests for the formal C/D runner, independent saved-result auditor, and zero-update safety guards."""
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("experiments", "tests"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import audit_copy_decoder_ffn_adaptation as auditor
import prepare_copy_decoder_ffn_adaptation as prep
import run_copy_decoder_ffn_adaptation as runner
from copy_decoder_ffn_adaptation import FFN_NAMES, check_order, restore, same
from copy_position_sites import digest, evaluate, forward, inputs, make_model, positions, sample, write
from prepare_copy_decoder_ffn_adaptation import STUDY_FILES, create_lock

MINI_COUNTS = {
    **{f"train_{n}": (8, 8) for n in range(3, 7)},
    **{f"validation_{n}": (2, 2) for n in range(3, 7)},
    **{f"test_{n}": (2, 2) for n in range(3, 7)},
    "test_8": (2, 2),
    "test_10": (0, 4),
}


def _copy_study_files(dst_root):
    for rel in STUDY_FILES:
        src = ROOT / rel
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _make_mini_pools():
    pools = {}
    for name, (n_dist, n_rep) in MINI_COUNTS.items():
        length = int(name.rsplit("_", 1)[1])
        dist = [[3 + (i + j) % 8 for j in range(length)] for i in range(n_dist)]
        rep = [[3 + (i % 8)] * length for i in range(n_rep)]
        pools[name] = dist + rep
    return pools


def _replay_unrestricted_segment(model, opt, gen, pg, train_pools, begin_step, stop_step, arm):
    stream = hashlib.sha256()
    pstream = hashlib.sha256()
    examples = 0
    targets_count = 0
    model.train()
    for step in range(begin_step + 1, stop_step + 1):
        n, idx, _ = sample(train_pools, gen, "mixed", step)
        pos = positions(n, pg)
        stream.update(json.dumps([n, idx.tolist()]).encode())
        pstream.update(json.dumps(pos.tolist()).encode())
        x = train_pools[f"train_{n}"][idx]
        prefix, y = inputs(x)
        opt.zero_grad(set_to_none=True)
        z = forward(model, x, prefix, pos if arm == "both" else None, pos)[0]
        F.cross_entropy(z.flatten(0, 1), y.flatten()).backward()
        opt.step()
        examples += len(x)
        targets_count += y.numel()
    cp = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(opt.state_dict()),
        step=stop_step,
        generator=gen.get_state().clone(),
        position_generator=pg.get_state().clone(),
    )
    return stream.hexdigest(), pstream.hexdigest(), examples, targets_count, cp


def _build_mini_study_workspace(root_dir):
    """Build an isolated mini archive with A/B continuation, protected hashes, and a valid lock bundle."""
    _copy_study_files(root_dir)
    data_pools = _make_mini_pools()
    data_path = root_dir / prep.DATA_REL
    data_path.parent.mkdir(parents=True, exist_ok=True)
    write(data_path, dict(pools=data_pools))

    torch.manual_seed(3767)
    base_model = make_model("rope")
    base_opt = torch.optim.Adam(base_model.parameters(), lr=0.001)
    # Warm Adam state across all parameters at step 6000.
    x_warm = torch.tensor(data_pools["train_3"][:2])
    p_warm, y_warm = inputs(x_warm)
    base_opt.zero_grad(set_to_none=True)
    F.cross_entropy(forward(base_model, x_warm, p_warm, None, None)[0].flatten(0, 1), y_warm.flatten()).backward()
    base_opt.step()
    for s in base_opt.state.values():
        s["step"] = torch.tensor(6000.0)

    start_cp = dict(
        model=copy.deepcopy(base_model.state_dict()),
        optimizer=copy.deepcopy(base_opt.state_dict()),
        step=6000,
        generator=torch.Generator().manual_seed(7001).get_state(),
        position_generator=torch.Generator().manual_seed(7002).get_state(),
    )
    start_path = root_dir / prep.START_REL
    start_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(start_cp, start_path)

    cont_dir = root_dir / prep.CONTINUATION_REL
    cont_dir.mkdir(parents=True, exist_ok=True)
    init_metrics, init_saved = {}, {}
    base_model.eval()
    for p_name, rows in data_pools.items():
        m, s = evaluate(base_model, rows, "cross_only")
        init_metrics[p_name] = m
        init_saved[p_name] = s
    init_scores_path = cont_dir / "3767_cross_only_6000_scores.pt"
    torch.save(init_saved, init_scores_path)

    train_pools = {k: torch.tensor(v) for k, v in data_pools.items() if k.startswith("train_")}
    ab_runs = []
    start_sha = digest(start_path)
    stage_examples = 210 + 168 + 140 + 120
    stage_targets = 4 * 840
    for arm in prep.ARMS:
        m_arm, o_arm, (g_arm, pg_arm) = restore(start_cp, lambda: make_model("rope"), frozen=False)
        st_hex, pst_hex, ex_n, tgt_n, cp_6004 = _replay_unrestricted_segment(
            m_arm, o_arm, g_arm, pg_arm, train_pools, 6000, 6004, arm
        )
        ckpt_p = cont_dir / f"3767_{arm}_6004.pt"
        torch.save(cp_6004, ckpt_p)
        ms, sv = {}, {}
        m_arm.eval()
        for p_name, rows in data_pools.items():
            m, s = evaluate(m_arm, rows, "cross_only")
            ms[p_name] = m
            sv[p_name] = s
        scores_p = cont_dir / f"3767_{arm}_6004_scores.pt"
        torch.save(sv, scores_p)
        ab_runs.append(dict(
            arm=arm,
            step=6004,
            start_sha256=start_sha,
            checkpoint=str(ckpt_p.relative_to(root_dir).as_posix()),
            checkpoint_sha256=digest(ckpt_p),
            stream=st_hex,
            position_stream=pst_hex,
            updates=4,
            examples=ex_n,
            active_targets=tgt_n,
            evaluation=dict(
                metrics=ms,
                scores=str(scores_p.relative_to(root_dir).as_posix()),
                scores_sha256=digest(scores_p),
                max_reference_error=0.0,
            ),
        ))

    ab_report_path = cont_dir / "report.json"
    ab_report = dict(
        status="complete",
        initial=dict(
            metrics=init_metrics,
            scores=str(init_scores_path.relative_to(root_dir).as_posix()),
            scores_sha256=digest(init_scores_path),
            max_reference_error=0.0,
        ),
        runs=ab_runs,
    )
    write(ab_report_path, ab_report)

    protected = {
        prep.START_REL: start_sha,
        prep.DATA_REL: digest(data_path),
        f"{prep.CONTINUATION_REL}/report.json": digest(ab_report_path),
    }
    stage_records = [
        dict(
            arm=arm,
            begin=step - prep.STAGE_UPDATES,
            step=step,
            updates=prep.STAGE_UPDATES,
            examples=prep.STAGE_EXAMPLES,
            targets=prep.STAGE_TARGETS,
            stream=f"stream_{step}",
            position_stream=f"pstream_{step}",
            bitwise_exact=True,
            max_abs_diff=0.0,
            seconds=0.1,
        )
        for step in prep.STAGES
        for arm in prep.ARMS
    ]
    env = dict(python="3.14", torch="test", threads=1, deterministic_algorithms=True, dtype="float64", device="cpu")
    fake_replay = dict(
        passed=True,
        formal_ready=False,
        formal_updates=0,
        replayed_arms=list(prep.ARMS),
        stages=list(prep.STAGES),
        replayed_updates_per_arm=len(prep.STAGES) * prep.STAGE_UPDATES,
        replayed_updates_total=len(prep.ARMS) * len(prep.STAGES) * prep.STAGE_UPDATES,
        bitwise_exact_all=True,
        eval_cells_checked=126,
        stage_records=stage_records,
        lineage_protected_files=len(protected),
        artifact_hashes=dict(protected),
        protected_hashes=dict(protected),
        environment=env,
    )
    fake_preflight = dict(
        passed=True,
        formal_ready=False,
        formal_updates=0,
        fixture_updates=17,
        profiles=[{}] * 16,
        audit_cells=10,
        checks=dict(saved_initialized_evaluator_and_audit=True),
        environment=dict(env),
    )
    create_lock(
        results_root=root_dir / "results",
        lock_name="copy_decoder_ffn_adaptation",
        root=root_dir,
        replay_fn=lambda _: fake_replay,
        preflight_fn=lambda: fake_preflight,
    )
    return dict(
        start_path=start_path,
        ab_report_path=ab_report_path,
        stage_examples=stage_examples,
        stage_targets=stage_targets,
    )


class FormalRunnerAndAuditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="copy-ffn-runner-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_authorization_output_guard_and_optimized_mode_rejected(self):
        with self.assertRaisesRegex(PermissionError, "authorize=True"):
            runner.run_formal("valid_dir", authorize=False, root=self.root)

        for reserved in ("copy_decoder_ffn_adaptation", "copy_exposure_continuation", "test_runs"):
            with self.subTest(reserved=reserved):
                with self.assertRaisesRegex(ValueError, "Reserved output directory"):
                    runner.run_formal(reserved, authorize=True, root=self.root)

        with self.assertRaisesRegex(ValueError, "alphanumeric"):
            runner.run_formal("../escape", authorize=True, root=self.root)

        if os.name == "nt":
            outside = self.root / "outside"
            outside.mkdir()
            results_dir = self.root / "results"
            proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(results_dir), str(outside)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.addCleanup(results_dir.rmdir)
            with self.assertRaisesRegex(ValueError, "reparse-point"):
                runner.run_formal("valid_out", authorize=True, root=self.root)

        for script in ("experiments/run_copy_decoder_ffn_adaptation.py", "experiments/audit_copy_decoder_ffn_adaptation.py"):
            with self.subTest(script=script):
                proc = subprocess.run([sys.executable, "-O", str(ROOT / script), "--help"],
                                      capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("optimized mode", proc.stderr)

    def test_preflight_exercises_initialized_audit_and_negative_controls(self):
        res = runner.runner_preflight(root=self.root, check_lock=False)
        self.assertTrue(res["passed"])
        self.assertFalse(res["formal_ready"])
        self.assertFalse(res["formal_authorized"])
        self.assertEqual(res["formal_updates"], 0)
        self.assertTrue(res["auditor_reloaded"])
        self.assertTrue(all(res["checks"].values()))

    def test_runtime_mismatch_rejected_before_training(self):
        _build_mini_study_workspace(self.root)
        with patch.object(runner.torch, "__version__", "0.0.0-mismatch"):
            with self.assertRaisesRegex(AssertionError, "locked historical A/B replay torch"):
                runner.run_formal(
                    "formal_cd_mismatch",
                    authorize=True,
                    root=self.root,
                    verify_runtime=True,
                    verify_history=False,
                )
        self.assertFalse((self.root / "results/formal_cd_mismatch/report.json").exists())

    def test_synthetic_cd_run_and_independent_audit_with_tamper_controls(self):
        meta = _build_mini_study_workspace(self.root)
        res = runner.run_formal(
            "copy_decoder_ffn_run_test",
            authorize=True,
            root=self.root,
            verify_runtime=False,
            verify_history=False,
            check_ab_checkpoints=True,
            expected_counts=MINI_COUNTS,
            expected_stages=(6004,),
            expected_stage_updates=4,
            expected_stage_examples=meta["stage_examples"],
            expected_stage_targets=meta["stage_targets"],
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["formal_updates"], 8)
        self.assertTrue(res["audit"]["passed"])
        self.assertEqual(res["audit"]["cd_updates"], 8)
        self.assertTrue(res["audit"]["frozen_ffn_bitwise_unchanged"])

        out_dir = self.root / "results/copy_decoder_ffn_run_test"
        self.assertTrue((out_dir / "audit.json").is_file())
        self.assertTrue((out_dir / "metrics.csv").is_file())

        # Independent reload re-audit without save=True passes.
        re_audit = auditor.audit_directory(
            out_dir,
            root=self.root,
            expected_counts=MINI_COUNTS,
            expected_stages=(6004,),
            expected_stage_updates=4,
            expected_stage_examples=meta["stage_examples"],
            expected_stage_targets=meta["stage_targets"],
            verify_lock=True,
            check_ab_checkpoints=True,
            save=False,
        )
        self.assertTrue(re_audit["passed"])

        # Tamper controls against the completed run directory.
        report_path = out_dir / "report.json"
        orig_report_bytes = report_path.read_bytes()
        csv_path = out_dir / "metrics.csv"
        orig_csv_bytes = csv_path.read_bytes()
        ckpt_c = out_dir / "3767_C_cross_only_6004.pt"
        orig_ckpt_c = ckpt_c.read_bytes()

        # 1. Corrupt frozen FFN parameter in saved C checkpoint + update SHA-256 -> caught!
        cp_bad_w = torch.load(ckpt_c, weights_only=True)
        with torch.no_grad():
            cp_bad_w["model"][FFN_NAMES[0]].add_(1e-6)
        torch.save(cp_bad_w, ckpt_c)
        rep = json.loads(orig_report_bytes)
        rep["runs"][0]["checkpoint_sha256"] = digest(ckpt_c)
        report_path.write_text(json.dumps(rep), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "Frozen decoder FFN parameter moved"):
            auditor.audit_directory(
                out_dir, root=self.root, expected_counts=MINI_COUNTS, expected_stages=(6004,),
                expected_stage_updates=4, expected_stage_examples=meta["stage_examples"],
                expected_stage_targets=meta["stage_targets"], verify_lock=True, save=False,
            )
        ckpt_c.write_bytes(orig_ckpt_c)
        report_path.write_bytes(orig_report_bytes)

        # 2. Increment frozen FFN Adam step in saved C checkpoint + update SHA-256 -> caught!
        cp_bad_opt = torch.load(ckpt_c, weights_only=True)
        start_snap, _ = auditor.capture_ffn_snapshot(torch.load(meta["start_path"], weights_only=True))
        pid = start_snap[FFN_NAMES[0]]["param_id"]
        cp_bad_opt["optimizer"]["state"][pid]["step"] = torch.tensor(6004.0)
        torch.save(cp_bad_opt, ckpt_c)
        rep = json.loads(orig_report_bytes)
        rep["runs"][0]["checkpoint_sha256"] = digest(ckpt_c)
        report_path.write_text(json.dumps(rep), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "Frozen decoder FFN Adam state moved"):
            auditor.audit_directory(
                out_dir, root=self.root, expected_counts=MINI_COUNTS, expected_stages=(6004,),
                expected_stage_updates=4, expected_stage_examples=meta["stage_examples"],
                expected_stage_targets=meta["stage_targets"], verify_lock=True, save=False,
            )
        ckpt_c.write_bytes(orig_ckpt_c)
        report_path.write_bytes(orig_report_bytes)

        # 3. Tamper metrics.csv + update metrics_sha256 -> caught by CSV reconciliation!
        csv_rows = list(csv.DictReader(orig_csv_bytes.decode("utf-8").splitlines(keepends=True)))
        csv_rows[0]["exact_accuracy"] = str(1.0 - float(csv_rows[0]["exact_accuracy"]))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        rep = json.loads(orig_report_bytes)
        rep["metrics_sha256"] = digest(csv_path)
        report_path.write_text(json.dumps(rep), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "Numeric mismatch"):
            auditor.audit_directory(
                out_dir, root=self.root, expected_counts=MINI_COUNTS, expected_stages=(6004,),
                expected_stage_updates=4, expected_stage_examples=meta["stage_examples"],
                expected_stage_targets=meta["stage_targets"], verify_lock=True, save=False,
            )
        csv_path.write_bytes(orig_csv_bytes)
        report_path.write_bytes(orig_report_bytes)

        # 4. Tamper decision summary -> caught!
        rep = json.loads(orig_report_bytes)
        rep["decision"]["primary"]["interaction"] = 99.0
        report_path.write_text(json.dumps(rep), encoding="utf-8")
        with self.assertRaises(AssertionError):
            auditor.audit_directory(
                out_dir, root=self.root, expected_counts=MINI_COUNTS, expected_stages=(6004,),
                expected_stage_updates=4, expected_stage_examples=meta["stage_examples"],
                expected_stage_targets=meta["stage_targets"], verify_lock=True, save=False,
            )
        report_path.write_bytes(orig_report_bytes)

        # 5. Tamper lock.json budget.decision_step -> caught before completion!
        lock_path = self.root / "results/copy_decoder_ffn_adaptation/lock.json"
        orig_lock_bytes = lock_path.read_bytes()
        bad_lock = json.loads(orig_lock_bytes)
        bad_lock["budget"]["decision_step"] = 7500
        lock_path.write_text(json.dumps(bad_lock), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "canonical protocol budget"):
            auditor.audit_directory(
                out_dir, root=self.root, expected_counts=MINI_COUNTS, expected_stages=(6004,),
                expected_stage_updates=4, expected_stage_examples=meta["stage_examples"],
                expected_stage_targets=meta["stage_targets"], verify_lock=True, save=False,
            )
        lock_path.write_bytes(orig_lock_bytes)


if __name__ == "__main__":
    unittest.main()
