from __future__ import annotations

import torch

from src.bank.core.probe import ProbeScorer


def test_probe_scorer_init():
    scorer = ProbeScorer(num_classes=100, smoothing=0.0)
    assert scorer.num_classes == 100
    assert len(scorer.smoothed_scores) == 100


def test_probe_scorer_update():
    scorer = ProbeScorer(num_classes=3, smoothing=0.0)
    scores = [0.5, 1.0, 2.0]
    smoothed = scorer.update(scores)

    assert smoothed == scores
    assert len(scorer.history) == 1


def test_probe_scorer_smoothing():
    scorer = ProbeScorer(num_classes=2, smoothing=0.9)
    smoothed = scorer.update([1.0, 2.0])

    assert abs(smoothed[0] - 0.1) < 1e-6
    assert abs(smoothed[1] - 0.2) < 1e-6

    smoothed2 = scorer.update([1.0, 2.0])
    assert abs(smoothed2[0] - 0.19) < 1e-6
    assert abs(smoothed2[1] - 0.38) < 1e-6


def test_normalised_scores():
    scorer = ProbeScorer(num_classes=4)
    scores = [1.0, 2.0, 4.0, 8.0]
    normed = scorer.normalised_scores(scores)

    assert normed[0] == 0.0
    assert normed[3] == 1.0
    assert all(0.0 <= n <= 1.0 for n in normed)


def test_normalised_scores_all_equal():
    scorer = ProbeScorer(num_classes=5)
    scores = [3.0, 3.0, 3.0, 3.0, 3.0]
    normed = scorer.normalised_scores(scores)

    assert all(n == 0.0 for n in normed)


def test_probe_loss():
    scorer = ProbeScorer(num_classes=5)

    class SimpleConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 5, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x).mean(dim=(2, 3))

    model = SimpleConvNet()

    probe_images = torch.randint(0, 256, (10, 32, 32, 3), dtype=torch.uint8)
    probe_targets = torch.zeros(10, dtype=torch.long)

    from src.data.cifar100.transforms import make_eval_transform
    transform = make_eval_transform(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    loss = scorer.compute_probe_loss(
        model, probe_images, probe_targets,
        class_id=0, device=torch.device("cpu"),
        transform=transform,
    )

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_prototype_probe_loss():
    scorer = ProbeScorer(num_classes=3)
    features = torch.randn(10, 64)
    prototype = torch.randn(64)

    loss = scorer.compute_prototype_probe_loss(features, prototype)

    assert isinstance(loss, float)
    assert 0.0 <= loss <= 2.0


def test_probe_scorer_state_dict():
    scorer = ProbeScorer(num_classes=3)
    scorer.update([0.1, 0.5, 0.9])

    state = scorer.state_dict()
    scorer2 = ProbeScorer(num_classes=3)
    scorer2.load_state_dict(state)

    assert scorer2.smoothed_scores == [0.1, 0.5, 0.9]
    assert len(scorer2.history) == 1
