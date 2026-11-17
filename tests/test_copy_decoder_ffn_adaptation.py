"""Explicit tests for the new helper contract and native initialized preflight."""
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
    from copy_decoder_ffn_adaptation import restore, snapshot, check_frozen, same, targets, preflight

    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Module(); self.base.decoder = nn.Module()
            self.base.decoder.linear1 = nn.Linear(3, 4, dtype=torch.float64)
            self.base.decoder.linear2 = nn.Linear(4, 3, dtype=torch.float64)
            self.head = nn.Linear(3, 2, dtype=torch.float64)

        def forward(self, x):
            d = self.base.decoder
            return self.head(x+d.linear2(torch.tanh(d.linear1(x))))


@unittest.skipUnless(HAS_TORCH, "PyTorch required; a skip is not a passing preflight")
class FreezeContract(unittest.TestCase):
    def setUp(self):
        self.context = torch.random.fork_rng(devices=[]); self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        torch.manual_seed(123)
        model = Fixture(); opt = torch.optim.Adam(model.parameters(), lr=.001)
        self.x = torch.arange(9, dtype=torch.float64).reshape(3, 3)/10
        model(self.x).square().sum().backward(); opt.step()
        self.cp = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
            generator=torch.Generator().manual_seed(31).get_state(),
            position_generator=torch.Generator().manual_seed(37).get_state())

    def test_frozen_state_and_input_gradients(self):
        original = copy.deepcopy(self.cp)
        model, opt, _ = restore(self.cp, Fixture, True)
        before = snapshot(model, opt)
        head = model.head.weight.detach().clone()
        x = self.x.clone().requires_grad_()
        opt.zero_grad(set_to_none=True); model(x).square().sum().backward()
        self.assertGreater(int(x.grad.count_nonzero()), 0)
        opt.step(); check_frozen(model, opt, before)
        self.assertFalse(torch.equal(head, model.head.weight))
        self.assertTrue(same(self.cp, original))

    def test_corrupt_parameter_and_adam_state_rejected(self):
        for corruption in ("parameter", "moment", "step"):
            with self.subTest(corruption=corruption):
                model, opt, _ = restore(self.cp, Fixture, True)
                before = snapshot(model, opt); parameter = next(iter(targets(model).values()))
                if corruption == "parameter":
                    with torch.no_grad(): parameter.add_(1)
                else:
                    opt.state[parameter]["exp_avg" if corruption == "moment" else "step"].add_(1)
                with self.assertRaises(AssertionError): check_frozen(model, opt, before)

    def test_unrestricted_restore_and_generators(self):
        model, opt, generators = restore(self.cp, Fixture, False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(same(opt.state_dict(), self.cp["optimizer"]))
        for generator, name in zip(generators, ("generator", "position_generator")):
            self.assertTrue(torch.equal(generator.get_state(), self.cp[name]))

    def test_zero_gradients_do_not_freeze_warmed_adam(self):
        model, opt, _ = restore(self.cp, Fixture, True)
        before = snapshot(model, opt)
        for parameter in targets(model).values():
            parameter.grad = torch.zeros_like(parameter)
        opt.step()
        self.assertTrue(any(not torch.equal(p.detach(), before[n]["parameter"]) for n, p in targets(model).items()))
        with self.assertRaises(AssertionError): check_frozen(model, opt, before)

    def test_missing_parameter_rejected(self):
        with self.assertRaises(AssertionError): targets(nn.Linear(3, 2))


@unittest.skipUnless(HAS_TORCH, "PyTorch required; native preflight not executed")
class NativePreflight(unittest.TestCase):
    def test_saved_initialized_model_and_audit(self):
        previous_threads = torch.get_num_threads()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.use_deterministic_algorithms, previous_determinism)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        report = preflight()
        self.assertTrue(report["passed"])
        self.assertFalse(report["formal_ready"])
        self.assertEqual(report["formal_updates"], 0)
        self.assertEqual(report["fixture_updates"], 17)
        self.assertEqual(len(report["profiles"]), 16)
        self.assertEqual(report["audit_cells"], 10)
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
