"""Synthetic regression tests for historical lineage, A/B replay, and exclusive lock creation."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("experiments", "tests"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from copy_decoder_ffn_adaptation import same
    from copy_position_sites import evaluate, forward, inputs, make_model
    import prepare_copy_decoder_ffn_adaptation as prep
    from prepare_copy_decoder_ffn_adaptation import (
        STUDY_FILES,
        create_lock,
        replay_ab,
        replay_segment,
        verify_historical_lineage,
        verify_pools,
        verify_protected_hashes,
        verify_start_checkpoint,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch required for historical verification and lock tests")
class HistoricalVerificationAndLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def _synthetic_pools(self):
        return {
            "train_3": [[3, 4, 5], [5, 4, 3], [3, 3, 4], [4, 4, 5]],
            "test_10": [[3] * 10, [4] * 10],
        }

    def _synthetic_counts(self):
        return {"train_3": (2, 2), "test_10": (0, 2)}

    def _synthetic_checkpoint(self, path, step=6000):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(77)
            model = make_model("rope")
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            x = torch.tensor([[3, 4, 5], [3, 3, 4]])
            prefix, y = inputs(x)
            loss = torch.nn.functional.cross_entropy(forward(model, x, prefix, None, None)[0].flatten(0, 1), y.flatten())
            loss.backward()
            opt.step()
            for state in opt.state.values():
                state["step"] = torch.tensor(float(step))
            payload = dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                step=step,
                generator=torch.Generator().manual_seed(101).get_state(),
                position_generator=torch.Generator().manual_seed(102).get_state(),
            )
            torch.save(payload, path)
            return payload

    def _write_study_files(self):
        for rel in STUDY_FILES:
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"synthetic study file")

    def _build_mini_archive(self, stages=(6004,)):
        """Create a minimal synthetic continuation archive matching the real artifact contract."""
        self._write_study_files()
        cont_dir = self.root / prep.CONTINUATION_REL
        cont_dir.mkdir(parents=True, exist_ok=True)
        start_path = self.root / prep.START_REL
        start_path.parent.mkdir(parents=True, exist_ok=True)
        start_cp = self._synthetic_checkpoint(start_path, step=6000)
        start_sha = hashlib.sha256(start_path.read_bytes()).hexdigest()

        pools = {
            f"train_{n}": [[3 + (i + j) % 8 for j in range(n)] for i in range(8)] + [[3 + i] * n for i in range(8)]
            for n in range(3, 7)
        }
        pools["test_10"] = [[3] * 10, [4] * 10]
        data_path = self.root / prep.DATA_REL
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(dict(pools=pools)), encoding="utf-8")
        data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()

        traj_audit_path = self.root / prep.TRAJECTORY_AUDIT_REL
        traj_audit_path.parent.mkdir(parents=True, exist_ok=True)
        traj_audit_path.write_text(json.dumps(dict(snapshot_artifact_hashes={prep.START_REL: start_sha})), encoding="utf-8")

        protected = {prep.START_REL: start_sha, prep.DATA_REL: data_sha}
        preflight_path = cont_dir / "preflight.json"
        preflight_path.write_text(json.dumps(dict(passed=True)), encoding="utf-8")
        plan_path = cont_dir / "plan.json"
        plan_path.write_text(json.dumps(dict(
            protected=protected,
            preflight_sha256=hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        )), encoding="utf-8")
        metrics_csv = cont_dir / "metrics.csv"
        metrics_csv.write_text("arm,step,pool,group\n", encoding="utf-8")

        train_tensors = {k: torch.tensor(v) for k, v in pools.items() if k.startswith("train_")}
        init_model, _, _ = prep.restore(start_cp, lambda: make_model("rope"), frozen=False)
        init_model.eval()
        init_metrics, init_scores = {}, {}
        for pname, rows in pools.items():
            init_metrics[pname], init_scores[pname] = evaluate(init_model, rows, "cross_only")
        init_scores_path = cont_dir / "3767_cross_only_6000_scores.pt"
        torch.save(init_scores, init_scores_path)

        runs = []
        prior_sha = {arm: start_sha for arm in prep.ARMS}
        arm_state = {}
        for arm in prep.ARMS:
            model, opt, (gen, pg) = prep.restore(start_cp, lambda: make_model("rope"), frozen=False)
            arm_state[arm] = (model, opt, gen, pg)
        prev = 6000
        for stop in stages:
            for arm in prep.ARMS:
                model, opt, gen, pg = arm_state[arm]
                s_hex, p_hex, ex, tg, ckpt = replay_segment(model, opt, gen, pg, train_tensors, prev, stop, arm)
                ckpt_path = cont_dir / f"3767_{arm}_{stop}.pt"
                torch.save(ckpt, ckpt_path)
                ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
                model.eval()
                eval_metrics, eval_scores = {}, {}
                for pname, rows in pools.items():
                    eval_metrics[pname], eval_scores[pname] = evaluate(model, rows, "cross_only")
                scores_path = cont_dir / f"3767_{arm}_{stop}_scores.pt"
                torch.save(eval_scores, scores_path)
                runs.append(dict(
                    arm=arm,
                    step=stop,
                    updates=stop - prev,
                    examples=ex,
                    active_targets=tg,
                    start_sha256=prior_sha[arm],
                    checkpoint=str(ckpt_path.relative_to(self.root).as_posix()),
                    checkpoint_sha256=ckpt_sha,
                    stream=s_hex,
                    position_stream=p_hex,
                    evaluation=dict(
                        metrics=eval_metrics,
                        scores=str(scores_path.relative_to(self.root).as_posix()),
                        scores_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                    ),
                ))
                prior_sha[arm] = ckpt_sha
            prev = stop

        report_path = cont_dir / "report.json"
        report_path.write_text(json.dumps(dict(
            status="complete",
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            metrics_sha256=hashlib.sha256(metrics_csv.read_bytes()).hexdigest(),
            initial=dict(
                metrics=init_metrics,
                scores=str(init_scores_path.relative_to(self.root).as_posix()),
                scores_sha256=hashlib.sha256(init_scores_path.read_bytes()).hexdigest(),
            ),
            runs=runs,
        )), encoding="utf-8")
        audit_path = cont_dir / "audit.json"
        audit_path.write_text(json.dumps(dict(
            passed=True,
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        )), encoding="utf-8")
        return cont_dir, report_path, audit_path

    def test_verify_pools_accepts_valid_and_rejects_corruptions(self):
        pools = self._synthetic_pools()
        counts = self._synthetic_counts()
        summary = verify_pools(pools, expected_counts=counts)
        self.assertEqual(summary, dict(pools=2, sequences=6, targets=38))

        swapped = copy.deepcopy(pools)
        swapped["train_3"][1], swapped["train_3"][2] = swapped["train_3"][2], swapped["train_3"][1]
        with self.assertRaisesRegex(AssertionError, "Pool order violation"):
            verify_pools(swapped, expected_counts=counts)

        bad_token = copy.deepcopy(pools)
        bad_token["train_3"][0][0] = 2
        with self.assertRaisesRegex(AssertionError, "Invalid token sequence"):
            verify_pools(bad_token, expected_counts=counts)

        bad_len = copy.deepcopy(pools)
        bad_len["test_10"].pop()
        with self.assertRaisesRegex(AssertionError, "Wrong row count"):
            verify_pools(bad_len, expected_counts=counts)

    def test_verify_start_checkpoint_accepts_valid_and_rejects_corruptions(self):
        ckpt_path = self.root / "start_6000.pt"
        self._synthetic_checkpoint(ckpt_path, step=6000)
        digest_hex = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

        summary = verify_start_checkpoint(ckpt_path, expected_sha256=digest_hex, expected_step=6000)
        self.assertEqual(summary["step"], 6000)
        self.assertEqual(summary["ffn_parameter_count"], 4)
        self.assertEqual(summary["trainable_when_frozen"], summary["parameter_count"] - 4)

        with self.assertRaisesRegex(AssertionError, "SHA-256 mismatch"):
            verify_start_checkpoint(ckpt_path, expected_sha256="0" * 64, expected_step=6000)

        bad_step_path = self.root / "bad_step.pt"
        self._synthetic_checkpoint(bad_step_path, step=5999)
        with self.assertRaisesRegex(AssertionError, "Unexpected start step"):
            verify_start_checkpoint(bad_step_path, expected_step=6000)

        bad_adam_path = self.root / "bad_adam.pt"
        payload = self._synthetic_checkpoint(bad_adam_path, step=6000)
        first_key = next(iter(payload["optimizer"]["state"]))
        payload["optimizer"]["state"][first_key]["step"] = torch.tensor(5999.0)
        torch.save(payload, bad_adam_path)
        with self.assertRaisesRegex(AssertionError, "Adam step mismatch"):
            verify_start_checkpoint(bad_adam_path, expected_step=6000)

        bad_keys_path = self.root / "bad_adam_keys.pt"
        payload_keys = self._synthetic_checkpoint(bad_keys_path, step=6000)
        moved = payload_keys["optimizer"]["state"].pop(0)
        payload_keys["optimizer"]["state"][999] = moved
        torch.save(payload_keys, bad_keys_path)
        with self.assertRaisesRegex(AssertionError, "Adam state keys mismatch"):
            verify_start_checkpoint(bad_keys_path, expected_step=6000)

        bad_shape_path = self.root / "bad_adam_shape.pt"
        payload_shape = self._synthetic_checkpoint(bad_shape_path, step=6000)
        payload_shape["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
        torch.save(payload_shape, bad_shape_path)
        with self.assertRaisesRegex(AssertionError, "Adam exp_avg tensor mismatch"):
            verify_start_checkpoint(bad_shape_path, expected_step=6000)

    def test_replay_segment_is_deterministic_and_rejects_invalid_inputs(self):
        pools = {
            f"train_{n}": torch.tensor([[3 + (i + j) % 8 for j in range(n)] for i in range(8)] + [[3 + i] * n for i in range(8)])
            for n in range(3, 7)
        }
        for arm in ("cross_only", "both"):
            with self.subTest(arm=arm):
                results = []
                for _ in range(2):
                    torch.manual_seed(17)
                    model = make_model("rope")
                    opt = torch.optim.Adam(model.parameters(), lr=0.001)
                    gen = torch.Generator().manual_seed(201)
                    pg = torch.Generator().manual_seed(202)
                    results.append(replay_segment(model, opt, gen, pg, pools, 6000, 6004, arm))
                self.assertEqual(results[0][:4], results[1][:4])
                self.assertTrue(same(results[0][4], results[1][4]))
                self.assertEqual(results[0][2], 210 + 168 + 140 + 120)
                self.assertEqual(results[0][3], 4 * 840)

        model = make_model("rope")
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        with self.assertRaisesRegex(AssertionError, "Unknown arm"):
            replay_segment(model, opt, torch.Generator(), torch.Generator(), pools, 6000, 6001, "self_only")
        with self.assertRaisesRegex(AssertionError, "stop_step must exceed begin_step"):
            replay_segment(model, opt, torch.Generator(), torch.Generator(), pools, 6001, 6001, "cross_only")

        real_sample = prep.sample

        def bad_sample(pool_map, generator, arm_name, step):
            n, idx, packed = real_sample(pool_map, generator, arm_name, step)
            torch.rand(1, generator=generator, dtype=torch.float64)
            return n, idx, packed

        with patch.object(prep, "sample", bad_sample):
            with self.assertRaisesRegex(AssertionError, "generator state mismatch"):
                replay_segment(model, opt, torch.Generator().manual_seed(5), torch.Generator().manual_seed(6), pools, 6000, 6001, "cross_only")

    def test_verify_historical_lineage_and_replay_ab_reject_tampering(self):
        cont_dir, report_path, audit_path = self._build_mini_archive(stages=(6004,))
        mini_counts = {f"train_{n}": (8, 8) for n in range(3, 7)}
        mini_counts["test_10"] = (0, 2)
        with patch.object(prep, "STAGES", (6004,)), \
             patch.object(prep, "STAGE_UPDATES", 4), \
             patch.object(prep, "STAGE_EXAMPLES", 210 + 168 + 140 + 120), \
             patch.object(prep, "STAGE_TARGETS", 4 * 840), \
             patch.object(prep, "validate_directory", return_value=dict(passed=True)), \
             patch.object(prep, "verify_pools", side_effect=lambda p: verify_pools(p, expected_counts=mini_counts)):
            lineage = verify_historical_lineage(self.root)
            self.assertTrue(lineage["passed"])
            replay = replay_ab(self.root, verify_eval=True)
            self.assertTrue(replay["passed"])
            self.assertTrue(replay["bitwise_exact_all"])
            self.assertEqual(replay["replayed_updates_total"], 8)

            original_report = report_path.read_text(encoding="utf-8")
            original_audit = audit_path.read_text(encoding="utf-8")

            def update_report_and_audit(mutator):
                data = json.loads(original_report)
                mutator(data)
                report_path.write_text(json.dumps(data), encoding="utf-8")
                audit_doc = json.loads(original_audit)
                audit_doc["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
                audit_path.write_text(json.dumps(audit_doc), encoding="utf-8")

            # 1. Tampered start_sha256 must fail lineage check.
            update_report_and_audit(lambda d: d["runs"][0].__setitem__("start_sha256", "0" * 64))
            with self.assertRaisesRegex(AssertionError, "start_sha256 mismatch"):
                verify_historical_lineage(self.root)

            # 2. Tampered stream hash must fail replay_ab.
            update_report_and_audit(lambda d: d["runs"][0].__setitem__("stream", "0" * 64))
            with self.assertRaisesRegex(AssertionError, "Stream hash mismatch"):
                replay_ab(self.root, verify_eval=False)

            # 3. Tampered position_stream hash must fail replay_ab.
            update_report_and_audit(lambda d: d["runs"][0].__setitem__("position_stream", "0" * 64))
            with self.assertRaisesRegex(AssertionError, "Position stream hash mismatch"):
                replay_ab(self.root, verify_eval=False)

            # 4. Tampered saved checkpoint must fail bitwise comparison in replay_ab.
            report_path.write_text(original_report, encoding="utf-8")
            audit_path.write_text(original_audit, encoding="utf-8")
            target_ckpt = cont_dir / "3767_cross_only_6004.pt"
            tampered_cp = torch.load(target_ckpt, weights_only=True)
            with torch.no_grad():
                first_param = next(iter(tampered_cp["model"]))
                tampered_cp["model"][first_param].add_(1.0)
            torch.save(tampered_cp, target_ckpt)
            new_sha = hashlib.sha256(target_ckpt.read_bytes()).hexdigest()
            update_report_and_audit(lambda d: d["runs"][0].__setitem__("checkpoint_sha256", new_sha))
            with self.assertRaisesRegex(AssertionError, "Bitwise checkpoint mismatch"):
                replay_ab(self.root, verify_eval=False)

    def test_create_lock_exclusive_and_protected_checks(self):
        self._write_study_files()
        protected = self.root / "results/protected.json"
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.write_bytes(b"{}")
        protected_sha = hashlib.sha256(protected.read_bytes()).hexdigest()

        fake_replay = dict(
            passed=True,
            bitwise_exact_all=True,
            eval_cells_checked=14,
            artifact_hashes={"results/protected.json": protected_sha},
            protected_hashes={"results/protected.json": protected_sha},
            environment=dict(python="3.14", torch="test", dtype="float64", device="cpu"),
        )
        fake_preflight = dict(passed=True, formal_updates=0, fixture_updates=17)

        res = create_lock(
            results_root=self.root / "results",
            lock_name="copy_decoder_ffn_adaptation",
            root=self.root,
            replay_fn=lambda _: fake_replay,
            preflight_fn=lambda: fake_preflight,
        )
        self.assertTrue(res["passed"])
        self.assertFalse(res["formal_ready"])
        self.assertFalse(res["formal_authorized"])
        self.assertEqual(res["formal_updates"], 0)
        lock_doc = json.loads((self.root / "results/copy_decoder_ffn_adaptation/lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock_doc["status"], "locked_awaiting_authorization")
        self.assertEqual(lock_doc["budget"]["quartet_max_updates"], 8000)

        with self.assertRaises(FileExistsError):
            create_lock(
                results_root=self.root / "results",
                lock_name="copy_decoder_ffn_adaptation",
                root=self.root,
                replay_fn=lambda _: fake_replay,
                preflight_fn=lambda: fake_preflight,
            )

        protected.write_bytes(b'{"tampered": true}')
        with self.assertRaisesRegex(AssertionError, "SHA-256 mismatch"):
            verify_protected_hashes(self.root, {"results/protected.json": protected_sha})

    def test_create_lock_rejects_no_eval(self):
        with self.assertRaisesRegex(AssertionError, "saved-score evaluation audit"):
            create_lock(
                results_root=self.root / "results",
                lock_name="copy_decoder_ffn_adaptation",
                root=self.root,
                verify_eval=False,
            )
        with self.assertRaises(SystemExit) as cm:
            prep.main(["lock", "--no-eval"])
        self.assertEqual(cm.exception.code, 2)

    @unittest.skipUnless(os.name == "nt", "Native Windows junction test")
    def test_create_lock_rejects_windows_junction_output(self):
        self._write_study_files()
        outside = self.root / "outside"
        outside.mkdir()
        results_dir = self.root / "results"
        proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(results_dir), str(outside)],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.addCleanup(results_dir.rmdir)
        with self.assertRaisesRegex(ValueError, "reparse-point"):
            create_lock(
                results_root=results_dir,
                lock_name="copy_decoder_ffn_adaptation",
                root=self.root,
                replay_fn=lambda _: dict(passed=True, bitwise_exact_all=True, eval_cells_checked=14,
                                          artifact_hashes={}, protected_hashes={}, environment={}),
                preflight_fn=lambda: dict(passed=True, formal_updates=0, fixture_updates=17),
            )
        self.assertEqual(list(outside.iterdir()), [])

    def test_create_lock_rejects_failed_replay_without_creating_dir(self):
        with self.assertRaisesRegex(AssertionError, "Exact A/B replay must pass"):
            create_lock(
                results_root=self.root / "results",
                lock_name="copy_decoder_ffn_adaptation",
                root=self.root,
                replay_fn=lambda _: dict(passed=False, bitwise_exact_all=False),
                preflight_fn=lambda: dict(passed=True, formal_updates=0),
            )
        self.assertFalse((self.root / "results/copy_decoder_ffn_adaptation").exists())


if __name__ == "__main__":
    unittest.main()
