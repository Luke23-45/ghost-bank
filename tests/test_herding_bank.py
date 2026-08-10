import torch

from src.methods.uniform_herding.herding import UniformHerdingReplayBank


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)


def _make_examples(labels: list[int]) -> list[tuple[torch.Tensor, int]]:
    examples: list[tuple[torch.Tensor, int]] = []
    for i, y in enumerate(labels):
        img = torch.full((3, 4, 4), float(i + 1))
        examples.append((img, y))
    return examples


def test_rebuild_selected():
    bank = UniformHerdingReplayBank(num_classes=3, total_budget=6, seed=0)
    bank.store(_make_examples([0, 0, 0, 1, 1, 1]))
    stats = bank.rebuild_selected(
        model=DummyModel(),
        allocation=[2, 2, 2],
        eval_transform=None,
    )

    assert stats["total"] == 4
    assert len(bank.selected[0]) == 2
    assert len(bank.selected[1]) == 2
    assert sum(len(pool) for pool in bank.selected.values()) == 4


def test_rebuild_selected_default_allocation_covers_expanded_classes():
    """``allocation=None`` must herd every class the bank currently knows.

    The bank is expanded as tasks arrive; defaulting to the *initial* class
    count would silently assign quota 0 to every later-introduced class and
    they would never be re-herded.
    """
    bank = UniformHerdingReplayBank(num_classes=4, total_budget=6, seed=0)
    bank.store(_make_examples([0, 0, 0, 1, 1, 1]))
    bank.expand(4)  # bank now knows classes 0..7
    for c in range(2, 8):
        bank.store(_make_examples([c, c]))

    stats = bank.rebuild_selected(
        model=DummyModel(),
        allocation=None,
        eval_transform=None,
    )

    assert set(bank.class_means.keys()) == set(range(8))
    assert stats["total"] == sum(len(pool) for pool in bank.selected.values())
    assert sum(len(pool) for pool in bank.selected.values()) == len(bank.selected)  # floor: 1 per class
