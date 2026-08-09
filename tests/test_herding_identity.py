"""Verification: each replay method uses its own herding implementation.

The pipeline must give ``icarl`` the canonical iCaRL bank (herd once at
arrival, truncate the ranked list afterwards) and ``uniform_herding`` the
proposed Uniform Herding bank (re-select every class in the current feature
space at every task boundary).  These tests run a two-task protocol with a
feature space that rotates between tasks so the two refresh policies must
behave observably differently.
"""

import torch

from src.methods.icarl.herding import iCaRLReplayBank
from src.methods.uniform_herding.herding import (
    UniformHerdingReplayBank,
    _herding_select,
)
from studies.runner.common.base_runner import create_bank

BUDGET = 6
TASK0_ALLOC = [3, 3, 0, 0]
TASK1_ALLOC = [2, 2, 1, 1]

# 90-degree rotation in the (x, z) plane.  Under it the mean of two feature
# vectors maps to the rotated mean, so re-selection in the new space can
# choose different exemplars than the original herding order.
ROTATION = torch.tensor(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
)


class _RotatingModel(torch.nn.Module):
    """Maps each raw image to a fixed feature row via its constant value,
    then applies ``rotation``.  The rotation simulates representation change
    between tasks."""

    def __init__(self, features: torch.Tensor, rotation: torch.Tensor) -> None:
        super().__init__()
        self._features = features
        self.rotation = rotation
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        idx = x[:, 0, 0, 0].long()
        return self._features[idx] @ self.rotation.T


def _identity_transform(batch: torch.Tensor) -> torch.Tensor:
    """Keep raw values intact so the model can recover each image's index."""
    return batch


def _feature_table() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(32, 3)


def _image(value: float) -> torch.Tensor:
    return torch.full((3, 4, 4), value)


def _identities(bank, class_id: int) -> list[int]:
    return [int(item[0][0, 0, 0].item()) for item in bank.selected[class_id]]


def _run_task0(bank, model) -> None:
    bank.store([(_image(float(i)), 0) for i in range(8)])
    bank.store([(_image(float(i)), 1) for i in range(8, 16)])
    bank.rebuild_selected(model, allocation=TASK0_ALLOC, eval_transform=_identity_transform)


def _run_task1(bank, model) -> None:
    bank.start_task()
    bank.store([(_image(float(i)), 2) for i in range(16, 24)])
    bank.store([(_image(float(i)), 3) for i in range(24, 32)])
    bank.rebuild_selected(model, allocation=TASK1_ALLOC, eval_transform=_identity_transform)


def test_task0_both_banks_use_greedy_herding_on_full_pool():
    features = _feature_table()
    for bank in (
        iCaRLReplayBank(num_classes=4, total_budget=BUDGET, seed=0),
        UniformHerdingReplayBank(num_classes=4, total_budget=BUDGET, seed=0),
    ):
        model = _RotatingModel(features, torch.eye(3))
        _run_task0(bank, model)
        expected = _herding_select(features[:8], 3)
        assert _identities(bank, 0) == expected, type(bank).__name__
        assert _identities(bank, 1) == [8 + i for i in _herding_select(features[8:16], 3)]


def test_icarl_old_class_is_truncated_not_reherded():
    features = _feature_table()
    bank = iCaRLReplayBank(num_classes=4, total_budget=BUDGET, seed=0)
    model = _RotatingModel(features, torch.eye(3))
    _run_task0(bank, model)
    task0_ranked = _identities(bank, 0)

    model.rotation = ROTATION
    _run_task1(bank, model)

    # Old class keeps the ranked prefix of the original order: the exemplar
    # identities never change after the class is herded.
    assert _identities(bank, 0) == task0_ranked[:2]
    # Class means are refreshed in the *current* feature space over the
    # retained exemplars (iCaRL Algorithm 1).
    kept = torch.stack([features[i] for i in task0_ranked[:2]]) @ ROTATION.T
    assert torch.allclose(bank.class_means[0], kept.mean(dim=0), atol=1e-6)
    # Fixed memory: total == M, transient pool discarded.
    assert sum(len(v) for v in bank.selected.values()) == BUDGET
    assert all(len(v) == 0 for v in bank._current_pool.values())


def test_uniform_herding_old_class_is_reselected_in_current_space():
    features = _feature_table()
    bank = UniformHerdingReplayBank(num_classes=4, total_budget=BUDGET, seed=0)
    model = _RotatingModel(features, torch.eye(3))
    _run_task0(bank, model)
    task0_ranked = _identities(bank, 0)

    model.rotation = ROTATION
    _run_task1(bank, model)

    # Old class is re-selected from its stored exemplars in the current
    # feature space, not kept as a fixed prefix.
    cands = torch.stack([features[i] for i in task0_ranked]) @ ROTATION.T
    expected_pick = _herding_select(cands, 2)
    assert _identities(bank, 0) == [task0_ranked[i] for i in expected_pick]
    # The refresh policy is observable: with this rotation the re-herded set
    # differs from the iCaRL truncation of the original ranking.
    assert _identities(bank, 0) != task0_ranked[:2]
    assert torch.allclose(bank.class_means[0], cands[expected_pick].mean(dim=0), atol=1e-6)
    assert sum(len(v) for v in bank.selected.values()) == BUDGET
    assert all(len(v) == 0 for v in bank._current_pool.values())


def test_icarl_new_classes_herded_once_from_full_pool():
    features = _feature_table()
    bank = iCaRLReplayBank(num_classes=4, total_budget=BUDGET, seed=0)
    model = _RotatingModel(features, torch.eye(3))
    _run_task0(bank, model)

    model.rotation = ROTATION
    _run_task1(bank, model)

    assert set(bank._herded) == {0, 1, 2, 3}
    # New classes are herded from their full current-task pool (one each).
    expected2 = _herding_select(features[16:24], 1)
    assert _identities(bank, 2) == [16 + expected2[0]]


def test_create_bank_dispatch_uses_dedicated_banks():
    from omegaconf import OmegaConf

    herding_cfg = OmegaConf.create({
        "data": {"memory_total": BUDGET},
        "bank": {"name": "uniform_herding", "floor": 1},
    })
    uh_bank = create_bank(herding_cfg, num_classes=4, run_seed=0)
    assert isinstance(uh_bank, UniformHerdingReplayBank)

    icarl_cfg = OmegaConf.create({
        "data": {"memory_total": BUDGET},
        "bank": {"name": "icarl", "floor": 1},
    })
    icarl_bank = create_bank(icarl_cfg, num_classes=4, run_seed=0)
    assert isinstance(icarl_bank, iCaRLReplayBank)
