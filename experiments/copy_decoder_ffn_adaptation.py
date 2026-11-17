"""Initialized-fixture preflight only; no historical continuation/run command."""
import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F

FFN_NAMES = tuple(
    f"base.decoder.{layer}.{field}"
    for layer in ("linear1", "linear2") for field in ("weight", "bias")
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def targets(model):
    named = dict(model.named_parameters())
    require(all(name in named for name in FFN_NAMES), "Missing decoder FFN parameter")
    result = {name: named[name] for name in FFN_NAMES}
    require(len({id(p) for p in result.values()}) == 4, "Aliased FFN parameters")
    return result


def restore(checkpoint, factory, frozen):
    require(type(frozen) is bool, "frozen must be bool")
    model = factory()
    model.load_state_dict(checkpoint["model"], strict=True)
    require(all(p.requires_grad for p in model.parameters()), "Factory must start fully trainable")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    for parameter in targets(model).values():
        parameter.requires_grad_(not frozen)
        parameter.grad = None
    generators = []
    for name in ("generator", "position_generator"):
        generator = torch.Generator()
        generator.set_state(checkpoint[name].clone())
        generators.append(generator)
    return model, optimizer, generators


def snapshot(model, optimizer):
    return {
        name: {"parameter": parameter.detach().clone(),
               "optimizer": copy.deepcopy(optimizer.state.get(parameter, {}))}
        for name, parameter in targets(model).items()
    }


def check_frozen(model, optimizer, before):
    require(set(before) == set(FFN_NAMES), "Incomplete frozen-state snapshot")
    for name, parameter in targets(model).items():
        require(not parameter.requires_grad and parameter.grad is None, "Frozen gradient contract violated")
        require(same(parameter.detach(), before[name]["parameter"]), "Frozen parameter moved")
        require(same(optimizer.state.get(parameter, {}), before[name]["optimizer"]), "Frozen Adam state moved")


def fixture_step(model, optimizer, content, pos, self_exposure, frozen):
    from copy_position_sites import forward, inputs
    prefix, labels = inputs(content)
    require(labels.tolist() == [row + [2] for row in content.tolist()], "Copy/EOS labels changed")
    require(prefix.tolist() == [[1] + row for row in content.tolist()], "BOS/prefix labels changed")
    before = snapshot(model, optimizer) if frozen else None
    optimizer.zero_grad(set_to_none=True)
    logits = forward(model, content, prefix, pos if self_exposure else None, pos)[0]
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    require(bool(torch.isfinite(loss)), "Nonfinite loss")
    loss.backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing/nonfinite gradient")
    optimizer.step()
    if frozen:
        check_frozen(model, optimizer, before)
    return float(loss.detach())


def preflight():
    from copy_position_sites import make_model, forward, inputs, evaluate, measures
    from audit_copy_position_sites import audit_cell
    from audit_copy_position_exposure import own_measures, close

    checks = {}; profiles = []; audit_cells = 0; max_error = 0.
    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory(prefix="copy-ffn-preflight-") as folder:
        torch.manual_seed(1499)
        factory = lambda: make_model("rope").eval()
        model = factory(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        rows = [[3, 4, 5], [3, 3, 4]]
        content = torch.tensor(rows); pos = torch.tensor([0, 3, 7, 10])
        path = Path(folder)/"initialized.pt"
        torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, weights_only=True))

        def audit_saved(candidate, tag):
            nonlocal audit_cells, max_error
            for name, pool in (("mixed", rows), ("missing_distinct", [[3]*10])):
                metrics, saved = evaluate(candidate, pool, "cross_only")
                artifact = Path(folder)/f"{tag}_{name}.pt"
                torch.save(saved, artifact)
                loaded = torch.load(artifact, weights_only=True)
                error, _ = audit_cell(candidate, pool, None, "cross_only", loaded, metrics)
                max_error = max(max_error, error); audit_cells += 1
                if name == "missing_distinct":
                    require(metrics["distinct"] is None, "Missing group must be null")
                else:
                    bad = copy.deepcopy(metrics); bad.pop("distinct")
                    try:
                        audit_cell(candidate, pool, None, "cross_only", loaded, bad)
                    except (AssertionError, KeyError):
                        pass
                    else:
                        raise AssertionError("Missing group accepted")
                    bad = copy.deepcopy(loaded)
                    bad["generated"][0] = [2] if loaded["generated"][0] != [2] else [3, 2]
                    try:
                        audit_cell(candidate, pool, None, "cross_only", bad, metrics)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("Corrupt generation accepted")
        audit_saved(model, "initialized")
        checks["saved_initialized_evaluator_and_audit"] = True

        labels = torch.tensor([row+[2] for row in rows])
        logits = torch.full((2, 4, 16), -8., dtype=torch.float64)
        logits.scatter_(-1, labels[..., None], 8.)
        scores = torch.zeros((2, 2, 4), dtype=torch.float64)
        for correct in (True, False):
            generated = [row+[2] for row in rows] if correct else [[2], [2]]
            metrics = measures(rows, logits, scores, generated)
            close(own_measures(rows, logits, scores, generated), metrics)
            require(metrics["all"]["exact_accuracy"] == int(correct), "Supplied generation mis-scored")
        checks["supplied_correct_wrong_generations"] = True

        fixture_step(model, optimizer, content, pos, False, False)
        checkpoint = dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                          generator=torch.Generator().manual_seed(2499).get_state(),
                          position_generator=torch.Generator().manual_seed(3499).get_state(), step=1)
        path = Path(folder)/"warmed_fixture.pt"; torch.save(checkpoint, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        original = copy.deepcopy(checkpoint)
        require(any(bool(state["exp_avg"].count_nonzero()) for state in optimizer.state.values()), "Adam fixture has no history")
        streams = []; updated_cells = 0
        for frozen in (False, True):
            for self_exposure in (False, True):
                candidate, opt, generators = restore(torch.load(path, weights_only=True), factory, frozen)
                require(all(p.requires_grad for n, p in candidate.named_parameters() if n not in FFN_NAMES), "Extra parameter frozen")
                hidden = torch.randn(2, 4, candidate.base.decoder.linear1.in_features, dtype=torch.float64, requires_grad=True)
                d = candidate.base.decoder
                value = d.linear2(F.relu(d.linear1(hidden)))
                gradient = torch.autograd.grad(value.square().sum(), hidden)[0]
                require(bool(torch.isfinite(gradient).all()) and bool(gradient.count_nonzero()), "FFN input gradients blocked")
                prefix, labels = inputs(content)
                trainable = tuple(p for p in candidate.parameters() if p.requires_grad)
                gradients = []; outputs = []
                for reference in (False, True):
                    z = forward(candidate, content, prefix, pos if self_exposure else None, pos, reference)[0]
                    gradients.append(torch.autograd.grad(F.cross_entropy(z.flatten(0, 1), labels.flatten()), trainable))
                    outputs.append(z.detach())
                torch.testing.assert_close(outputs[0], outputs[1], atol=1e-10, rtol=0)
                for a, b in zip(*gradients):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=0)
                stream = []
                for n in range(3, 7):
                    batch = 840//(n+1)
                    x = torch.randint(3, 11, (batch, n), generator=generators[0])
                    sampled = torch.randperm(11, generator=generators[1])[:n+1].sort().values
                    stream.append((x.tolist(), sampled.tolist()))
                    before = copy.deepcopy(candidate.state_dict())
                    tick = time.perf_counter()
                    loss = fixture_step(candidate, opt, x, sampled, self_exposure, frozen)
                    elapsed = time.perf_counter()-tick
                    require(any(not torch.equal(v, before[k]) for k, v in candidate.state_dict().items() if k not in FFN_NAMES), "No unfrozen parameter updated")
                    profiles.append(dict(frozen=frozen, self_exposure=self_exposure, length=n, batch=batch,
                        targets=batch*(n+1), loss=loss, step_seconds=elapsed,
                        parameter_bytes=sum(p.numel()*p.element_size() for p in candidate.parameters()),
                        optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))))
                    updated_cells += 1
                streams.append(stream)
                audit_saved(candidate, f"frozen{int(frozen)}_self{int(self_exposure)}")
        require(all(stream == streams[0] for stream in streams), "Fixture streams differ")
        require(same(checkpoint, original) and hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Fixture start mutated")
        checks.update(reference_outputs_and_trainable_gradients=True, frozen_parameters_and_adam_unchanged=True,
                      input_gradients_preserved=True, other_parameters_update=True, fixture_streams_matched=True)
    return dict(passed=True, formal_ready=False, checks=checks, audit_cells=audit_cells, max_reference_error=max_error,
                fixture_updates=1+updated_cells, profiles=profiles, formal_updates=0,
                cpu_peak_process_bytes=None, gpu_allocated_bytes=None, gpu_reserved_bytes=None,
                limitation="Initialized fixtures only; fixture streams are not historical sampler replay. CPU peak memory and native-Windows execution still require measurement. No trained-checkpoint replay, historical hashes, outcome criterion, or formal-study approval is supplied.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight",))
    parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(preflight(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
