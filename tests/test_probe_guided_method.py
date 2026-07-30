from __future__ import annotations

import torch

from src.methods.probe_guided.method import ProbeGuidedMethod


def test_method_init():
    method = ProbeGuidedMethod(
        retrieval_budget=32,
        memory_total=2000,
        gamma=0.5,
        beta=1.0,
    )
    assert method.retrieval_budget == 32
    assert method.memory_total == 2000
    assert method.gamma == 0.5


def test_method_update_allocation():
    method = ProbeGuidedMethod(memory_total=100, memory_floor=1)
    method._exemplar_bank = {0: [(1, 0)], 1: [(2, 1)], 2: [(3, 2)]}

    probe_scores = [10.0, 1.0, 0.1]
    method.update_allocation(probe_scores)

    assert method._current_allocation is not None
    assert len(method._current_allocation) == 3
    assert sum(method._current_allocation) == 100
    assert all(a >= 1 for a in method._current_allocation)


def test_method_prune_memory():
    method = ProbeGuidedMethod(memory_total=50, memory_floor=1)
    method._exemplar_bank = {
        0: [("a", 0), ("b", 0), ("c", 0)],
        1: [("d", 1), ("e", 1)],
        2: [("f", 2), ("g", 2), ("h", 2), ("i", 2)],
    }
    method._current_allocation = [1, 1, 2]

    method.prune_memory()

    assert len(method._exemplar_bank[0]) == 1
    assert len(method._exemplar_bank[1]) == 1
    assert len(method._exemplar_bank[2]) == 2


def test_method_prune_memory_removes_class():
    method = ProbeGuidedMethod(memory_total=10, memory_floor=0)
    method._exemplar_bank = {0: [("a", 0)], 1: [("b", 1)]}
    method._current_allocation = [1, 0]

    method.prune_memory()

    assert 0 in method._exemplar_bank
    assert 1 not in method._exemplar_bank


def test_method_store_exemplars():
    method = ProbeGuidedMethod()
    method._store_exemplars([(torch.tensor([1, 2, 3]), 0), (torch.tensor([4, 5, 6]), 1)])

    assert 0 in method._exemplar_bank
    assert 1 in method._exemplar_bank
    assert len(method._exemplar_bank[0]) == 1
    assert len(method._exemplar_bank[1]) == 1


def test_method_store_exemplars_dedups_per_task_indices():
    method = ProbeGuidedMethod()
    raw = [(torch.tensor([1, 2, 3]), 0), (torch.tensor([4, 5, 6]), 1)]
    indices = torch.tensor([7, 8], dtype=torch.long)

    method._store_exemplars(raw, raw_indices=indices)
    method._store_exemplars(raw, raw_indices=indices)

    assert len(method._exemplar_bank[0]) == 1
    assert len(method._exemplar_bank[1]) == 1


def test_method_set_replay_class_count_clears_seen_indices():
    method = ProbeGuidedMethod()
    raw = [(torch.tensor([1, 2, 3]), 0)]
    indices = torch.tensor([3], dtype=torch.long)

    method.set_replay_class_count(10)
    method._store_exemplars(raw, raw_indices=indices)
    method._store_exemplars(raw, raw_indices=indices)
    assert len(method._exemplar_bank[0]) == 1

    method.set_replay_class_count(20)
    method._store_exemplars(raw, raw_indices=indices)
    assert len(method._exemplar_bank[0]) == 2


def test_method_state_dict_roundtrip():
    method = ProbeGuidedMethod(memory_total=100)
    method._exemplar_bank = {0: [(1, 0)], 1: [(2, 1)]}
    method._current_allocation = [50, 50]

    state = method.state_dict()
    method2 = ProbeGuidedMethod(memory_total=100)
    method2.load_state_dict(state)

    assert method2._current_allocation == [50, 50]
    assert len(method2._exemplar_bank) == 2


def test_method_snapshot_model():
    method = ProbeGuidedMethod()

    class FakePLModule:
        def __init__(self):
            self.model = torch.nn.Linear(10, 5)

    pl_module = FakePLModule()
    method.snapshot_model(pl_module)

    assert method._prev_model_snapshot is not pl_module.model
    assert isinstance(method._prev_model_snapshot, torch.nn.Module)
    assert next(method._prev_model_snapshot.parameters()).requires_grad is False


def test_method_replay_class_count_gates_new_classes():
    method = ProbeGuidedMethod(retrieval_budget=4, memory_total=10)
    method._exemplar_bank = {
        0: [(torch.randn(3, 32, 32), 0), (torch.randn(3, 32, 32), 0)],
        1: [(torch.randn(3, 32, 32), 1), (torch.randn(3, 32, 32), 1)],
        2: [(torch.randn(3, 32, 32), 2), (torch.randn(3, 32, 32), 2)],
    }
    method.set_replay_class_count(2)

    assert method._replay_class_count == 2
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 3))
    loss = method._compute_replay_loss(model, torch.device("cpu"))
    assert loss.item() >= 0.0


def test_method_update_allocation_no_scores():
    method = ProbeGuidedMethod(memory_total=100)
    method._exemplar_bank = {0: [(1, 0)], 1: [(2, 1)]}

    method.update_allocation(None)
    assert method._current_allocation is not None
    assert sum(method._current_allocation) == 100


def test_method_replay_uses_context_transform():
    method = ProbeGuidedMethod(retrieval_budget=1, memory_total=10)
    raw = torch.full((32, 32, 3), 7, dtype=torch.uint8)
    method._exemplar_bank = {0: [(raw, 0)]}
    method.set_replay_class_count(1)

    captured: dict[str, torch.Tensor] = {}

    def transform(x: torch.Tensor) -> torch.Tensor:
        captured["input"] = x.clone()
        return torch.ones_like(x, dtype=torch.float32)

    class TinyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            captured["replay_x"] = x.clone()
            return torch.zeros(x.size(0), 1, device=x.device)

    loss = method._compute_replay_loss(
        TinyModel(),
        torch.device("cpu"),
        transform=transform,
        rng=torch.Generator().manual_seed(0),
    )

    assert loss.item() >= 0.0
    assert "input" in captured
    assert "replay_x" in captured
    assert captured["input"].shape == (3, 32, 32)
    assert torch.allclose(captured["replay_x"], torch.ones(1, 3, 32, 32))
