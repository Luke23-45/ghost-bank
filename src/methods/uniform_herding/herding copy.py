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
    """Greedy herding that approximates the class mean."""
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


class UniformHerdingReplayBank(AbstractGhostBank):
    """Fixed-total replay bank that re-selects exemplars with herding.

    The active bank contains only the selected exemplars from completed tasks.
    Examples from the current task are held in a transient candidate pool until
    the task-boundary rebuild; that pool is discarded immediately afterward.

    This is the proposed method's memory: after every task, the exemplar set of
    every observed class is rebuilt in the current feature space.  Old classes
    are re-selected from their currently stored exemplars combined with the
    transient current-task pool; the pool is discarded after selection, so
    persistent replay storage never exceeds the fixed budget ``M``.
    """

    def __init__(
        self,
        num_classes: int,
        total_budget: int,
        seed: int,
        floor: int = 1,
        exclude_classes: Collection[int] | None = None,
        selection: str = "herding",
    ) -> None:
        if selection not in ("herding", "random"):
            raise ValueError(
                f"selection must be 'herding' or 'random', got {selection!r}"
            )
        excluded = set(exclude_classes) if exclude_classes is not None else set()
        self._bank: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._selected: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._current_pool: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self.class_means: dict[int, torch.Tensor] = {}
        self._seen_indices: set[int] = set()
        self._rng = random.Random(seed)
        self._total_budget = total_budget
        self._floor = floor
        self._num_classes = num_classes
        self._selection = selection

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
        return sample_uniform(self._selected, budget, self._rng)

    def expand(self, num_new_classes: int) -> None:
        max_existing = max(self._bank.keys()) if self._bank else -1
        for c in range(max_existing + 1, max_existing + 1 + num_new_classes):
            if c not in self._bank:
                self._bank[c] = []
            if c not in self._selected:
                self._selected[c] = []
            if c not in self._current_pool:
                self._current_pool[c] = []

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
            # they would never be selected.
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
            print(f"[rebuild_selected] Starting. allocation len={len(allocation)}, bank keys={sorted(self._bank.keys())}", flush=True)
            for cid in self._bank:
                print(
                    f"  class {cid}: active={len(self._bank[cid])} "
                    f"current={len(self._current_pool.get(cid, []))}",
                    flush=True,
                )

        with torch.inference_mode():
            for class_id, quota in enumerate(allocation):
                pool = self._bank.get(class_id, []) + self._current_pool.get(class_id, [])
                if quota <= 0 or not pool:
                    continue
                classes_used += 1
                t_class = time.time()
                feats_chunks: list[torch.Tensor] = []
                for start in range(0, len(pool), chunk_size):
                    end = min(start + chunk_size, len(pool))
                    t_chunk = time.time()
                    raw_batch = torch.stack([_to_chw(item[0]) for item in pool[start:end]], dim=0)
                    t_stack = time.time()
                    images_t = _transform_raw_batch(raw_batch, eval_transform).to(model_device)
                    t_transform = time.time()
                    feats = model.extract_features(images_t).detach().cpu()
                    t_feat = time.time()
                    feats_chunks.append(feats)
                    if verbose:
                        print(f"  class {class_id}: chunk [{start}:{end}] stack={t_stack-t_chunk:.3f}s transform={t_transform-t_stack:.3f}s extract={t_feat-t_transform:.3f}s", flush=True)

                feats_all = torch.cat(feats_chunks, dim=0)

                if quota >= len(pool):
                    pick = list(range(len(pool)))
                    if verbose:
                        print(f"  class {class_id}: quota={quota} >= pool={len(pool)}, copied. ({time.time()-t_class:.2f}s)", flush=True)
                elif self._selection == "random":
                    pick = self._rng.sample(range(len(pool)), quota)
                    if verbose:
                        print(f"  class {class_id}: pool={len(pool)} quota={quota} random-selection ({time.time()-t_class:.2f}s)", flush=True)
                else:
                    t_herd = time.time()
                    pick = _herding_select(feats_all, quota)
                    if verbose:
                        print(f"  class {class_id}: pool={len(pool)} quota={quota} herding={time.time()-t_herd:.3f}s total={time.time()-t_class:.2f}s", flush=True)
                
                # NME should use the mean of the *selected* exemplars
                class_mean = feats_all[pick].mean(dim=0)
                self.class_means[class_id] = class_mean.cpu()
                
                selected[class_id] = [pool[i] for i in pick]
                total_selected += len(selected[class_id])
        if verbose:
            print(f"[rebuild_selected] Done in {time.time()-t_rebuild_start:.2f}s. total_selected={total_selected}", flush=True)

        # Only the selected exemplars survive the task boundary. The current
        # task pool is transient and must not become an unbounded archive.
        self._bank = selected
        self._selected = selected
        self._current_pool = {c: [] for c in selected}
        return {
            "classes": classes_used,
            "total": total_selected,
            "min": min((len(v) for v in selected.values()), default=0),
            "max": max((len(v) for v in selected.values()), default=0),
            "mean": float(total_selected / max(1, classes_used)),
        }

    @property
    def selected(self) -> dict[int, list]:
        return self._selected

    def state_dict(self) -> dict:
        return {
            "bank": {c: list(pool) for c, pool in self._bank.items()},
            "selected": {c: list(pool) for c, pool in self._selected.items()},
            "current_pool": {c: list(pool) for c, pool in self._current_pool.items()},
            "class_means": {c: mean.clone() for c, mean in self.class_means.items()},
            "seen_indices": list(self._seen_indices),
            "total_budget": self._total_budget,
            "floor": self._floor,
        }

    def load_state_dict(self, state: dict) -> None:
        bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        selected = {int(c): list(pool) for c, pool in state.get("selected", {}).items()}
        current_pool = {
            int(c): list(pool)
            for c, pool in state.get("current_pool", {}).items()
        }
        self.class_means = {
            int(c): mean.clone()
            for c, mean in state.get("class_means", {}).items()
        }
        self._seen_indices = set(int(i) for i in state.get("seen_indices", []))
        self._selected = selected
        if "current_pool" in state:
            self._bank = bank
            self._current_pool = current_pool
        elif selected:
            # Older checkpoints stored the unbounded candidate archive in
            # ``bank``. Resume them from their bounded selected state.
            self._bank = selected
            self._current_pool = {c: [] for c in selected}
        else:
            self._bank = bank
            self._current_pool = {c: [] for c in bank}
        self._total_budget = state.get("total_budget", self._total_budget)
        self._floor = state.get("floor", self._floor)
