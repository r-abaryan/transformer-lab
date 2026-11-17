"""Explicit regression tests; no historical training/checkpoint execution."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from copy_decoder_ffn_adaptation import (restore, snapshot, check_frozen, same, targets,
        check_order, sampler_fixture, process_memory, preflight)

    class Fixture(nn.Module):
        def __init__(self):
            super().__init__(); self.base = nn.Module(); self.base.decoder = nn.Module()
            self.base.decoder.linear1 = nn.Linear(3, 4, dtype=torch.float64)
            self.base.decoder.linear2 = nn.Linear(4, 3, dtype=torch.float64)
            self.head = nn.Linear(3, 2, dtype=torch.float64)

        def forward(self, x):
            d = self.base.decoder
            return self.head(x+d.linear2(torch.tanh(d.linear1(x))))


@unittest.skipUnless(HAS_TORCH, "PyTorch required; a skip is not preflight success")
class FreezeContract(unittest.TestCase):
    def setUp(self):
        context = torch.random.fork_rng(devices=[]); context.__enter__()
        self.addCleanup(context.__exit__, None, None, None); torch.manual_seed(123)
        model = Fixture(); opt = torch.optim.Adam(model.parameters(), lr=.001)
        self.x = torch.arange(9, dtype=torch.float64).reshape(3, 3)/10
        model(self.x).square().sum().backward(); opt.step()
        self.cp = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
            generator=torch.Generator().manual_seed(31).get_state(), position_generator=torch.Generator().manual_seed(37).get_state())

    def test_frozen_state_and_input_gradients(self):
        original = copy.deepcopy(self.cp); model, opt, _ = restore(self.cp, Fixture, True)
        before = snapshot(model, opt); head = model.head.weight.detach().clone(); x = self.x.clone().requires_grad_()
        self.assertTrue(all(not p.requires_grad for p in targets(model).values()))
        self.assertTrue(all(p.requires_grad for n, p in model.named_parameters() if n not in before))
        opt.zero_grad(set_to_none=True); model(x).square().sum().backward()
        self.assertGreater(int(x.grad.count_nonzero()), 0); opt.step(); check_frozen(model, opt, before)
        self.assertFalse(torch.equal(head, model.head.weight)); self.assertTrue(same(self.cp, original))

    def test_corrupt_parameter_and_adam_state_rejected(self):
        for name in targets(Fixture()):
            for corruption in ("parameter", "exp_avg", "exp_avg_sq", "step"):
                with self.subTest(parameter=name, corruption=corruption):
                    model, opt, _ = restore(self.cp, Fixture, True); before = snapshot(model, opt); p = targets(model)[name]
                    if corruption == "parameter":
                        with torch.no_grad(): p.add_(1)
                    else:
                        opt.state[p][corruption].add_(1)
                    with self.assertRaises(AssertionError): check_frozen(model, opt, before)

    def test_unrestricted_restore_and_generators(self):
        model, opt, generators = restore(self.cp, Fixture, False)
        self.assertTrue(all(p.requires_grad for p in model.parameters())); self.assertTrue(same(opt.state_dict(), self.cp["optimizer"]))
        for g, name in zip(generators, ("generator", "position_generator")):
            self.assertTrue(torch.equal(g.get_state(), self.cp[name]))

    def test_zero_gradients_do_not_freeze_warmed_adam(self):
        model, opt, _ = restore(self.cp, Fixture, True); before = snapshot(model, opt)
        for p in targets(model).values(): p.grad = torch.zeros_like(p)
        opt.step()
        self.assertTrue(any(not same(p.detach(), before[n]["parameter"]) for n, p in targets(model).items()))
        with self.assertRaises(AssertionError): check_frozen(model, opt, before)

    def test_optimizer_order_perturbations_rejected(self):
        model, opt, _ = restore(self.cp, Fixture, True)
        opt.param_groups[0]["params"].reverse()
        with self.assertRaisesRegex(AssertionError, "order changed"): check_order(model, opt)
        bad = copy.deepcopy(self.cp); bad["optimizer"]["param_groups"][0]["params"].reverse()
        with self.assertRaisesRegex(AssertionError, "Saved optimizer ordering"): restore(bad, Fixture, True)

    def test_same_distinguishes_signed_zero(self):
        a = torch.tensor([0.], dtype=torch.float64); b = torch.tensor([-0.], dtype=torch.float64)
        self.assertTrue(torch.equal(a, b)); self.assertFalse(same(a, b))
        self.assertTrue(same(torch.tensor(1.), torch.tensor(1.)))

    def test_missing_parameter_rejected(self):
        with self.assertRaises(AssertionError): targets(nn.Linear(3, 2))


@unittest.skipUnless(HAS_TORCH, "PyTorch required; native checks not executed")
class NativePreflight(unittest.TestCase):
    def test_historical_sampler_on_synthetic_pools(self):
        result = sampler_fixture(); self.assertTrue(result["passed"]); self.assertEqual(result["sampler_draw_shape"], [2, 105])
        self.assertTrue(result["midstream_restore"])

    def test_extra_sampler_rng_consumption_rejected(self):
        from copy_position_sites import sample
        def wrong(pools, generator, arm, step):
            result = sample(pools, generator, arm, step)
            torch.rand(1, generator=generator, dtype=torch.float64)
            return result
        with self.assertRaisesRegex(AssertionError, "RNG consumption changed"): sampler_fixture(wrong)

    def test_process_memory_contract(self):
        result = process_memory()
        if sys.platform == "win32":
            self.assertTrue(result["available"])
            for key in ("peak_working_set_bytes", "working_set_bytes", "private_commit_bytes", "peak_private_commit_bytes"):
                self.assertGreater(result[key], 0)
        else:
            self.assertFalse(result["available"])

    def test_saved_initialized_model_and_audit(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled())
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True); report = preflight()
        self.assertTrue(report["passed"]); self.assertFalse(report["formal_ready"]); self.assertEqual(report["formal_updates"], 0)
        self.assertEqual(report["fixture_updates"], 17); self.assertEqual(len(report["profiles"]), 16); self.assertEqual(report["audit_cells"], 10)
        self.assertTrue(all(report["checks"].values())); self.assertEqual(report["largest_batch"], 210)
        self.assertEqual(len(report["gradient_checks"]), 2)
        for result in report["gradient_checks"]:
            self.assertTrue(result["detachment_rejected"]); self.assertTrue(result["detached_forward_values_preserved"])
            self.assertLessEqual(result["max_shared_gradient_error"], 1e-10)
        if sys.platform == "win32":
            before, after = report["memory"]["before"], report["memory"]["after"]
            self.assertGreaterEqual(after["peak_working_set_bytes"], before["peak_working_set_bytes"])
            self.assertGreaterEqual(after["peak_private_commit_bytes"], before["peak_private_commit_bytes"])
            self.assertEqual(report["cpu_peak_process_bytes"], after["peak_working_set_bytes"])


if __name__ == "__main__":
    unittest.main()
