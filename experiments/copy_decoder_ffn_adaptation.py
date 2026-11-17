"""Initialized-fixture validation only; no historical continuation command."""
import argparse
import copy
import ctypes
import hashlib
import json
import platform
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F

FFN_NAMES = tuple(f"base.decoder.{layer}.{field}" for layer in ("linear1", "linear2") for field in ("weight", "bias"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(operation):
    try:
        operation()
    except AssertionError:
        return True
    raise AssertionError("Negative control was accepted")


def same(a, b):
    if torch.is_tensor(a):
        return (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def targets(model):
    named = dict(model.named_parameters())
    require(all(n in named for n in FFN_NAMES), "Missing decoder FFN parameter")
    result = {n: named[n] for n in FFN_NAMES}
    require(len({id(p) for p in result.values()}) == 4, "Aliased FFN parameters")
    return result


def check_order(model, optimizer):
    require(len(optimizer.param_groups) == 1, "Historical Adam uses one parameter group")
    require([id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in model.parameters()], "Optimizer parameter order changed")


def restore(checkpoint, factory, frozen):
    require(type(frozen) is bool, "frozen must be bool")
    model = factory(); model.load_state_dict(checkpoint["model"], strict=True)
    require(all(p.requires_grad for p in model.parameters()), "Factory must start fully trainable")
    groups = checkpoint["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["params"] == list(range(len(tuple(model.parameters())))), "Saved optimizer ordering violates historical contract")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    check_order(model, optimizer)
    for p in targets(model).values():
        p.requires_grad_(not frozen); p.grad = None
    generators = []
    for name in ("generator", "position_generator"):
        g = torch.Generator(); g.set_state(checkpoint[name].clone()); generators.append(g)
    return model, optimizer, generators


def snapshot(model, optimizer):
    return {n: dict(parameter=p.detach().clone(), optimizer=copy.deepcopy(optimizer.state.get(p, {}))) for n, p in targets(model).items()}


def check_frozen(model, optimizer, before):
    require(set(before) == set(FFN_NAMES), "Incomplete frozen snapshot")
    check_order(model, optimizer)
    for n, p in targets(model).items():
        require(not p.requires_grad and p.grad is None, "Frozen gradient must be None")
        require(same(p.detach(), before[n]["parameter"]), "Frozen parameter moved")
        require(same(optimizer.state.get(p, {}), before[n]["optimizer"]), "Frozen Adam state moved")


def process_memory():
    if sys.platform != "win32":
        return dict(available=False, platform=sys.platform, method=None, reason="Windows GetProcessMemoryInfo unavailable")
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    current = kernel.GetCurrentProcess; current.argtypes = []; current.restype = wintypes.HANDLE
    read = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]; read.restype = wintypes.BOOL
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    if not read(current(), ctypes.byref(counters), counters.cb):
        return dict(available=False, platform=sys.platform, method="GetProcessMemoryInfo", error=ctypes.get_last_error())
    return dict(available=True, platform=sys.platform, method="GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX",
        peak_working_set_bytes=int(counters.PeakWorkingSetSize), working_set_bytes=int(counters.WorkingSetSize),
        private_commit_bytes=int(counters.PrivateUsage), peak_private_commit_bytes=int(counters.PeakPagefileUsage),
        scope="Process-lifetime peaks, including imports, optimizer and evaluation; not an isolated-step increment")


def sampler_fixture(sampler=None):
    from copy_position_sites import sample, positions
    sampler = sample if sampler is None else sampler
    pools = {f"train_{n}": torch.tensor([[3+(i+j)%8 for j in range(n)] for i in range(8)] + [[3+i]*n for i in range(8)]) for n in range(3, 7)}
    g = torch.Generator().manual_seed(4241); pg = torch.Generator().manual_seed(4243)
    rg = torch.Generator(); rg.set_state(g.get_state()); rpg = torch.Generator(); rpg.set_state(pg.get_state())
    stream = hashlib.sha256(); reference = hashlib.sha256(); pstream = hashlib.sha256(); pref = hashlib.sha256()
    for j, step in enumerate(range(6001, 6005)):
        if j == 2:
            gs, ps = g.get_state().clone(), pg.get_state().clone()
            g = torch.Generator(); g.set_state(gs); pg = torch.Generator(); pg.set_state(ps)
        n, idx, packed = sampler(pools, g, "mixed", step); pos = positions(n, pg)
        expected_n = 3+(step-1)%4; batch = 840//(expected_n+1)
        ranks = torch.rand(2, 105, generator=rg, dtype=torch.float64)
        expected = [int(float(ranks[h, k])*8)+8*h for h in range(2) for k in range(batch//2)]
        expected_pos = torch.randperm(11, generator=rpg)[:expected_n+1].sort().values.tolist()
        require(n == expected_n and idx.tolist() == expected and pos.tolist() == expected_pos, "Historical sampler fixture mismatch")
        require(packed == struct.pack("210d", *ranks.flatten().tolist()), "Fixed rank draw changed")
        require(torch.equal(g.get_state(), rg.get_state()) and torch.equal(pg.get_state(), rpg.get_state()), "Sampler RNG consumption changed")
        stream.update(json.dumps([n, idx.tolist()]).encode()); reference.update(json.dumps([expected_n, expected]).encode())
        pstream.update(json.dumps(pos.tolist()).encode()); pref.update(json.dumps(expected_pos).encode())
        rejects(lambda: require(idx.flip(0).tolist() == expected, "Reordered sample indices"))
        rejects(lambda: require(pos.flip(0).tolist() == expected_pos, "Reversed position order"))
    require(stream.digest() == reference.digest() and pstream.digest() == pref.digest(), "Stream encoding changed")
    return dict(passed=True, updates=4, sampler_draw_shape=[2, 105], stream_sha256=stream.hexdigest(),
        position_stream_sha256=pstream.hexdigest(), midstream_restore=True, reordered_indices_rejected=True,
        reversed_positions_rejected=True, limitation="Synthetic pools and fixture generators only; not historical checkpoint/pool or A/B trajectory replay")


def full_forward_check(checkpoint, factory, content, pos, self_exposure):
    from copy_position_sites import forward, inputs
    observations = []
    for frozen, detach in ((False, False), (True, False), (True, True)):
        model, optimizer, _ = restore(checkpoint, factory, frozen); captured = {}
        def hook(module, args, output):
            output.retain_grad(); captured["branch"] = output
            return output.detach() if detach else output
        handle = model.base.decoder.linear2.register_forward_hook(hook)
        try:
            prefix, labels = inputs(content)
            z = forward(model, content, prefix, pos if self_exposure else None, pos)[0]
            F.cross_entropy(z.flatten(0, 1), labels.flatten()).backward()
        finally:
            handle.remove()
        gradients = {}
        for n, p in model.named_parameters():
            if n in FFN_NAMES and frozen:
                require(p.grad is None, "Frozen FFN received a gradient")
            elif n not in FFN_NAMES:
                require(p.grad is not None and bool(torch.isfinite(p.grad).all()), "Missing shared gradient")
                gradients[n] = p.grad.detach().clone()
        branch = captured["branch"].grad
        if not detach:
            require(branch is not None and bool(branch.count_nonzero()), "No loss gradient on full-forward FFN output")
        else:
            require(branch is None, "Detachment negative control ineffective")
        require(same(model.state_dict(), checkpoint["model"]), "Gradient check changed model state")
        observations.append((z.detach(), gradients))
    def compare(a, b):
        torch.testing.assert_close(a[0], b[0], atol=1e-10, rtol=0)
        require(a[1].keys() == b[1].keys(), "Shared gradient set changed")
        for n in a[1]:
            torch.testing.assert_close(a[1][n], b[1][n], atol=1e-10, rtol=0)
    compare(observations[0], observations[1])
    torch.testing.assert_close(observations[1][0], observations[2][0], atol=1e-10, rtol=0)
    upstream = [n for n in observations[1][1] if not n.startswith(("base.output.", "base.decoder.norm3."))]
    require(any(bool(observations[1][1][n].count_nonzero()) for n in upstream), "No upstream learning signal")
    require(any(not torch.allclose(observations[1][1][n], observations[2][1][n], atol=1e-10, rtol=0) for n in upstream), "Detached branch did not change upstream gradients")
    rejects(lambda: compare(observations[1], observations[2]))
    return dict(passed=True, self_exposure=self_exposure, detached_forward_values_preserved=True,
        detachment_rejected=True, shared_parameters=len(observations[1][1]),
        max_shared_gradient_error=max(float((observations[0][1][n]-observations[1][1][n]).abs().max()) for n in observations[0][1]))


def fixture_step(model, optimizer, content, pos, self_exposure, frozen):
    from copy_position_sites import forward, inputs
    check_order(model, optimizer); prefix, labels = inputs(content)
    require(labels.tolist() == [r+[2] for r in content.tolist()] and prefix.tolist() == [[1]+r for r in content.tolist()], "Copy/EOS labels changed")
    before = snapshot(model, optimizer) if frozen else None
    optimizer.zero_grad(set_to_none=True)
    z = forward(model, content, prefix, pos if self_exposure else None, pos)[0]
    loss = F.cross_entropy(z.flatten(0, 1), labels.flatten()); require(bool(torch.isfinite(loss)), "Nonfinite loss")
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad), "Missing/nonfinite gradient")
    optimizer.step()
    if frozen:
        check_frozen(model, optimizer, before)
    return float(loss.detach())


def preflight():
    tick = time.perf_counter(); memory_before = process_memory()
    from copy_position_sites import make_model, forward, inputs, evaluate, measures
    from audit_copy_position_sites import audit_cell
    from audit_copy_position_exposure import own_measures, close
    profiles = []; cells = 0; maximum = 0.
    sampler = sampler_fixture()
    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix="copy-ffn-preflight-") as folder:
        torch.manual_seed(1499); factory = lambda: make_model("rope").eval()
        model = factory(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rows = [[3, 4, 5], [3, 3, 4]]; content = torch.tensor(rows); pos = torch.tensor([0, 3, 7, 10])
        path = Path(folder)/"initialized.pt"; torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, weights_only=True))
        def audit_saved(candidate, tag):
            nonlocal cells, maximum
            for name, pool in (("mixed", rows), ("missing_distinct", [[3]*10])):
                metrics, saved = evaluate(candidate, pool, "cross_only")
                artifact = Path(folder)/f"{tag}_{name}.pt"; torch.save(saved, artifact)
                loaded = torch.load(artifact, weights_only=True)
                error, _ = audit_cell(candidate, pool, None, "cross_only", loaded, metrics)
                maximum = max(maximum, error); cells += 1
                if name == "missing_distinct":
                    require(metrics["distinct"] is None, "Absent group must be null")
                else:
                    bad = copy.deepcopy(metrics); bad.pop("distinct")
                    try:
                        audit_cell(candidate, pool, None, "cross_only", loaded, bad)
                    except (AssertionError, KeyError):
                        pass
                    else:
                        raise AssertionError("Missing group accepted")
                    bad = copy.deepcopy(loaded); bad["generated"][0] = [2] if loaded["generated"][0] != [2] else [3, 2]
                    rejects(lambda: audit_cell(candidate, pool, None, "cross_only", bad, metrics))
        audit_saved(model, "initialized")
        labels = torch.tensor([r+[2] for r in rows]); logits = torch.full((2, 4, 16), -8., dtype=torch.float64)
        logits.scatter_(-1, labels[..., None], 8.); scores = torch.zeros((2, 2, 4), dtype=torch.float64)
        for correct in (True, False):
            generated = [r+[2] for r in rows] if correct else [[2], [2]]
            metrics = measures(rows, logits, scores, generated); close(own_measures(rows, logits, scores, generated), metrics)
            require(metrics["all"]["exact_accuracy"] == int(correct), "Supplied generation mis-scored")
        fixture_step(model, optimizer, content, pos, False, False); updates = 1
        checkpoint = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
            generator=torch.Generator().manual_seed(2499).get_state(), position_generator=torch.Generator().manual_seed(3499).get_state(), step=1)
        require(any(bool(s["exp_avg"].count_nonzero()) for s in optimizer.state.values()), "Adam fixture has no history")
        path = Path(folder)/"warmed.pt"; torch.save(checkpoint, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest(); original = copy.deepcopy(checkpoint)
        gradient_checks = [full_forward_check(checkpoint, factory, content, pos, exposure) for exposure in (False, True)]
        streams = []
        for frozen in (False, True):
            for exposure in (False, True):
                candidate, opt, generators = restore(torch.load(path, weights_only=True), factory, frozen)
                require(all(p.requires_grad for n, p in candidate.named_parameters() if n not in FFN_NAMES), "Extra parameter frozen")
                prefix, y = inputs(content); trainable = tuple(p for p in candidate.parameters() if p.requires_grad); gradients = []; outputs = []
                for reference in (False, True):
                    z = forward(candidate, content, prefix, pos if exposure else None, pos, reference)[0]
                    gradients.append(torch.autograd.grad(F.cross_entropy(z.flatten(0, 1), y.flatten()), trainable)); outputs.append(z.detach())
                torch.testing.assert_close(outputs[0], outputs[1], atol=1e-10, rtol=0)
                for a, b in zip(*gradients):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=0)
                stream = []
                for n in range(3, 7):
                    batch = 840//(n+1); x = torch.randint(3, 11, (batch, n), generator=generators[0])
                    sampled = torch.randperm(11, generator=generators[1])[:n+1].sort().values; stream.append((x.tolist(), sampled.tolist()))
                    before = copy.deepcopy(candidate.state_dict()); start = time.perf_counter()
                    loss = fixture_step(candidate, opt, x, sampled, exposure, frozen); updates += 1; elapsed = time.perf_counter()-start
                    require(any(not same(v, before[k]) for k, v in candidate.state_dict().items() if k not in FFN_NAMES), "No unfrozen update")
                    profiles.append(dict(frozen=frozen, self_exposure=exposure, length=n, batch=batch, targets=batch*(n+1), loss=loss, step_seconds=elapsed,
                        parameter_bytes=sum(p.numel()*p.element_size() for p in candidate.parameters()),
                        optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))))
                streams.append(stream); audit_saved(candidate, f"{int(frozen)}_{int(exposure)}")
        require(all(s == streams[0] for s in streams), "Fixture streams differ")
        require(same(checkpoint, original) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Fixture start mutated")
    memory_after = process_memory()
    if sys.platform == "win32":
        require(memory_before["available"] and memory_after["available"] and memory_after["peak_working_set_bytes"] > 0, "Windows memory measurement failed")
    return dict(passed=True, formal_ready=False, formal_updates=0, fixture_updates=updates, profiles=profiles, audit_cells=cells, max_reference_error=maximum,
        checks=dict(saved_initialized_evaluator_and_audit=True, supplied_correct_wrong_generations=True, reference_outputs_and_trainable_gradients=True,
            frozen_parameters_and_adam_unchanged=True, input_gradients_preserved=True, other_parameters_update=True, fixture_streams_matched=True,
            full_forward_detachment_rejected=True, optimizer_order_preserved=True, historical_sampler_fixture=True), gradient_checks=gradient_checks, sampler_fixture=sampler,
        memory=dict(before=memory_before, after=memory_after), cpu_peak_process_bytes=memory_after.get("peak_working_set_bytes"),
        gpu_allocated_bytes=None, gpu_reserved_bytes=None, largest_batch=max(p["batch"] for p in profiles), seconds=time.perf_counter()-tick,
        environment=dict(python=sys.version, torch=torch.__version__, platform=platform.platform(), cpu=platform.processor(),
            threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), dtype="float64", device="cpu"),
        limitation="Initialized fixtures and shared historical model helpers; not saved-pool/checkpoint A/B replay. Peaks cover this process lifetime, not a formal-run upper bound. No formal authorization.")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("preflight",)); parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    print(json.dumps(preflight(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
