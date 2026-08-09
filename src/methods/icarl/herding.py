from __future__ import annotations

import random
import time
from collections.abc import Collection

import torch

from src.bank.core.allocator import allocate_uniform_fixed_total
from src.bank.core.base import AbstractGhostBank, _to_int
from src.bank.core.retrieval import sample_uniform


def _to_chw(raw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    if raw.dim() == 3 and raw.shape[-1] == 3 and raw.shape[0] != 3:
        return raw.permute(2, 0, 1).contiguous()
    return raw.contiguous()


def _transform_raw_batch(raw_images: torch.Tensor, transform) -> torch.Tensor:
    if raw_images.dim() != 4:
        raise ValueError(f"Expected a 4D batch, got shape {tuple(raw_images.shape)}")
    if raw_images.shape[-1] == 3 and raw_images.shape[1] != 3:
        batch = raw_images.permute(0, 3, 1, 2).contiguous()
    else:
        batch = raw_images.contiguous()
    if transform is not None:
        return transform(batch)
    return batch.float().div(255.0)


def _herding_select(features: torch.Tensor, budget: int) -> list[int]:
    """Greedy herding that approximates the class mean (iCaRL Algorithm 4).

    Returns the prioritized selection order ``[p_1, ..., p_budget]``, where
    ``p_k`` minimizes ``||mu - (1/k)[phi(x) + sum_{j<k} phi(p_j)]||``.  Earlier
    elements are more important: any prefix of the returned order is the
    canonical exemplar set for that size.
    """
    n = features.shape[0]
    if n == 0 or budget <= 0:
        return []
    budget = min(budget, n)
    class_mean = features.mean(dim=0)
    selected: list[int] = []
    selected_sum = torch.zeros_like(class_mean)

    available = torch.ones(n, dtype=torch.bool)

    for k in range(1, budget + 1):
        target = k * class_mean - selected_sum
        dists = torch.sum((features - target) ** 2, dim=1)
        dists[~available] = float("inf")
        best_idx = int(torch.argmin(dists).item())

        selected.append(best_idx)
        available[best_idx] = False
        selected_sum += features[best_idx]

    return selected


class iCaRLReplayBank(AbstractGhostBank):
    """Canonical iCaRL exemplar memory: herd once on arrival, truncate later.

    When a class is first observed, its exemplar set is constructed once by
    herding over the full class data in the current feature space (iCaRL
    Algorithm 4) and stored as a prioritized list.  At later task boundaries,
    old classes are never re-herded from data: the ranked list is reduced to
    the new quota by keeping the first ``quota`` exemplars (iCaRL Algorithm
    5).  Class means for NME classification are recomputed each task in the
    current feature space over the retained exemplars (iCaRL Algorithm 1).

    This differs from the Uniform Herding bank in the refresh policy only:
    the greedy herding rule is shared, but old-class exemplars are selected
    exactly once and only truncated afterwards.  Persistent storage never
    exceeds the fixed budget ``M`` at any task boundary.
    """

    def __init__(
        self,
        num_classes: int,
        total_budget: int,
        seed: int,
        floor: int = 1,
        exclude_classes: Collection[int] | None = None,
    ) -> None:
        excluded = set(exclude_classes) if exclude_classes is not None else set()
        self._bank: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._current_pool: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._herded: set[int] = set()
        self.class_means: dict[int, torch.Tensor] = {}
        self._seen_indices: set[int] = set()
        self._rng = random.Random(seed)
        self._total_budget = total_budget
        self._floor = floor
        self._num_classes = num_classes

    @staticmethod
    def _to_tensor_label(y: object) -> torch.Tensor:
        if torch.is_tensor(y):
            return y
        return torch.tensor(y, dtype=torch.long)

    def start_task(self) -> None:
        self._seen_indices.clear()
        self._current_pool = {c: [] for c in self._bank}

    def store(self, examples: list, raw_indices: torch.Tensor | None = None) -> None:
        indices = raw_indices.tolist() if raw_indices is not None else None
        for pos, (x, y) in enumerate(examples):
            if indices is not None:
                sample_idx = int(indices[pos])
                if sample_idx in self._seen_indices:
                    continue
                self._seen_indices.add(sample_idx)
            y = self._to_tensor_label(y)
            cid = _to_int(y)
            if cid in self._current_pool:
                self._current_pool[cid].append((x, y))

    def query(self, budget: int, **kwargs) -> list:
        if budget <= 0:
            return []
        return sample_uniform(self._bank, budget, self._rng)

    def expand(self, num_new_classes: int) -> None:
        max_existing = max(self._bank.keys()) if self._bank else -1
        for c in range(max_existing + 1, max_existing + 1 + num_new_classes):
            if c not in self._bank:
                self._bank[c] = []
            if c not in self._current_pool:
                self._current_pool[c] = []

    @property
    def selected(self) -> dict[int, list]:
        return self._bank

    def rebuild_selected(
        self,
        model,
        allocation: list[int] | None = None,
        eval_transform=None,
        device: torch.device | None = None,
        chunk_size: int = 256,
        verbose: bool = False,
    ) -> dict[str, float]:
        if allocation is None:
            # Default to the *currently known* classes (the bank is expanded
            # as tasks arrive).  Allocating over the initial ``num_classes``
            # would silently give every later-introduced class quota 0, so
            # they would never be herded.
            allocation = allocate_uniform_fixed_total(
                num_classes=len(self._bank),
                total_budget=self._total_budget,
                floor=self._floor,
            )

        selected: dict[int, list] = {c: [] for c in self._bank.keys()}
        total_selected = 0
        classes_used = 0
        model.eval()
        model_device = next(model.parameters()).device
        t_rebuild_start = time.time()
        if verbose:
            print(f"[iCaRL rebuild_selected] Starting. allocation len={len(allocation)}, herded={sorted(self._herded)}", flush=True)

        def _features(pool: list) -> torch.Tensor:
            feats_chunks: list[torch.Tensor] = []
            for start in range(0, len(pool), chunk_size):
                end = min(start + chunk_size, len(pool))
                raw_batch = torch.stack([_to_chw(item[0]) for item in pool[start:end]], dim=0)
                images_t = _transform_raw_batch(raw_batch, eval_transform).to(model_device)
                feats_chunks.append(model.extract_features(images_t).detach().cpu())
            return torch.cat(feats_chunks, dim=0) if feats_chunks else torch.empty(0)

        with torch.inference_mode():
            for class_id, quota in enumerate(allocation):
                if quota <= 0:
                    continue

                if class_id in self._herded:
                    # Old class: never re-herd from data.  Keep the first
                    # ``quota`` exemplars of the ranked list (Algorithm 5),
                    # then recompute the class mean over the retained
                    # exemplars in the current feature space (Algorithm 1).
                    exemplars = self._bank.get(class_id, [])[:quota]
                    selected[class_id] = list(exemplars)
                    if exemplars:
                        classes_used += 1
                        total_selected += len(exemplars)
                        feats = _features(exemplars)
                        self.class_means[class_id] = feats.mean(dim=0).cpu()
                    continue

                # New class: construct the exemplar set once from the full
                # current-task pool in the current feature space (Algorithm 4).
                pool = self._current_pool.get(class_id, [])
                if not pool:
                    continue
                classes_used += 1
                feats_all = _features(pool)
                pick = _herding_select(feats_all, quota)
                exemplars = [pool[i] for i in pick]
                selected[class_id] = exemplars
                total_selected += len(exemplars)
                self.class_means[class_id] = feats_all[pick].mean(dim=0).cpu()
                self._herded.add(class_id)

        if verbose:
            print(f"[iCaRL rebuild_selected] Done in {time.time()-t_rebuild_start:.2f}s. total_selected={total_selected}", flush=True)

        # Only the selected exemplars survive the task boundary.  The current
        # task pool is transient and must not become an unbounded archive.
        self._bank = selected
        self._current_pool = {c: [] for c in selected}
        return {
            "classes": classes_used,
            "total": total_selected,
            "min": min((len(v) for v in selected.values()), default=0),
            "max": max((len(v) for v in selected.values()), default=0),
            "mean": float(total_selected / max(1, classes_used)),
        }

    def state_dict(self) -> dict:
        return {
            "bank": {c: list(pool) for c, pool in self._bank.items()},
            "current_pool": {c: list(pool) for c, pool in self._current_pool.items()},
            "herded": sorted(self._herded),
            "total_budget": self._total_budget,
            "floor": self._floor,
        }

    def load_state_dict(self, state: dict) -> None:
        self._bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        self._current_pool = {
            int(c): list(pool)
            for c, pool in state.get("current_pool", {}).items()
        }
        self._herded = set(int(c) for c in state.get("herded", []))
        self._total_budget = state.get("total_budget", self._total_budget)
        self._floor = state.get("floor", self._floor)
