from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


class ProbeScorer:
    """Computes and tracks held-out probe loss per class.

    For each class ``c``, the probe score at task ``t`` is:

        s_c(t) = (1 / |P_c|) * sum_{x in P_c} CE(f_t(x), c)

    where ``P_c`` is the held-out probe set for class ``c``.
    Scores are stored in a history dict for analysis.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.0) -> None:
        self.num_classes = num_classes
        self.smoothing = smoothing
        self._history: list[list[float]] = []
        self._smoothed_scores: list[float] = [0.0] * num_classes
        self._raw_scores: list[float] = [0.0] * num_classes

    def compute_probe_loss(
        self,
        model: torch.nn.Module,
        probe_images: torch.Tensor,
        probe_targets: torch.Tensor,
        class_id: int,
        device: torch.device,
        transform: object | None = None,
    ) -> float:
        """Compute the probe loss for a single class.

        Parameters
        ----------
        model : nn.Module
            Current model (in eval mode).
        probe_images : torch.Tensor
            Raw NHWC uint8 images for this class ``[N, H, W, C]``.
        probe_targets : torch.Tensor
            Labels ``[N]``.
        class_id : int
            Class ID (for distillation-style masking if needed).
        device : torch.device
            Target device.
        transform : callable or None
            Optional transform (eval transform) applied before forward.

        Returns
        -------
        float
            Mean cross-entropy loss over the probe set.
        """
        targets = probe_targets.to(device)
        images = probe_images.to(device)

        if transform is not None:
            batch_list = []
            for i in range(images.shape[0]):
                img_nhwc = images[i]
                img_nchw = img_nhwc.permute(2, 0, 1).contiguous()
                batch_list.append(transform(img_nchw))
            images_t = torch.stack(batch_list, dim=0)
        else:
            images_t = images.float() / 255.0
            if images_t.dim() == 4 and images_t.shape[-1] == 3 and images_t.shape[1] != 3:
                images_t = images_t.permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            logits = model(images_t)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        return loss.item()

    def compute_prototype_probe_loss(
        self,
        features: torch.Tensor,
        prototype: torch.Tensor,
    ) -> float:
        """Compute probe score as cosine distance from class prototype.

        For PTM-based methods with prototype classifiers:

            s_c(t) = (1 / |P_c|) * sum [1 - cos(phi(x), mu_c)]

        where ``mu_c`` is the class prototype.
        """
        features_norm = F.normalize(features, dim=-1)
        proto_norm = F.normalize(prototype.unsqueeze(0), dim=-1)
        cos_sim = torch.mm(features_norm, proto_norm.t()).squeeze(-1)
        loss = (1.0 - cos_sim).mean().item()
        return loss

    def update(
        self,
        scores: list[float],
    ) -> list[float]:
        """Ingest new per-class probe scores, update smoothed state.

        Parameters
        ----------
        scores : list[float]
            Probe loss for each class ``c`` (length = num_classes).
            Classes not yet seen may have score 0.0.

        Returns
        -------
        list[float]
            Smoothed probe scores.
        """
        self._raw_scores = list(scores)
        self._history.append(list(scores))

        if self.smoothing > 0.0:
            for c in range(self.num_classes):
                self._smoothed_scores[c] = (
                    self.smoothing * self._smoothed_scores[c]
                    + (1.0 - self.smoothing) * scores[c]
                )
        else:
            self._smoothed_scores = list(scores)

        return self.smoothed_scores

    def normalised_scores(self, scores: list[float] | None = None) -> list[float]:
        """Return scores normalised to [0, 1] via min-max scaling.

        A score of 1.0 means highest forgetting (most important to replay).
        When all scores are equal, returns zeros.
        """
        vals = scores if scores is not None else self._smoothed_scores
        min_v = min(vals)
        max_v = max(vals)
        if max_v - min_v < 1e-12:
            return [0.0] * len(vals)
        return [(v - min_v) / (max_v - min_v) for v in vals]

    @property
    def smoothed_scores(self) -> list[float]:
        return list(self._smoothed_scores)

    @property
    def raw_scores(self) -> list[float]:
        return list(self._raw_scores)

    @property
    def history(self) -> list[list[float]]:
        return [list(h) for h in self._history]

    def state_dict(self) -> dict:
        return {
            "history": [list(h) for h in self._history],
            "smoothed_scores": list(self._smoothed_scores),
            "raw_scores": list(self._raw_scores),
        }

    def load_state_dict(self, state: dict) -> None:
        self._history = [list(h) for h in state["history"]]
        self._smoothed_scores = list(state["smoothed_scores"])
        self._raw_scores = list(state["raw_scores"])
