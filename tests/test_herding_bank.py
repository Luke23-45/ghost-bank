import torch

from src.bank.strategies.herding import HerdingReplayBank


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


def test_rebuild_selected_and_query():
    bank = HerdingReplayBank(num_classes=3, total_budget=6, seed=0)
    bank.store(_make_examples([0, 0, 0, 1, 1, 1]))
    stats = bank.rebuild_selected(
        model=DummyModel(),
        allocation=[2, 2, 2],
        eval_transform=None,
    )

    assert stats["total"] == 4
    assert len(bank.selected[0]) == 2
    assert len(bank.selected[1]) == 2
    assert bank.query(3)
