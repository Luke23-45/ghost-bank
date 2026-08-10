from __future__ import annotations

import random
import time
from collections.abc import Collection

import torch

from src.bank.core.allocator import allocate_uniform_fixed_total
from src.bank.core.base import AbstractGhostBank, _to_int


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
    """Fixed-memory replay bank that re-selects exemplars with herding.

    Two nested, both-bounded budgets:
      - `total_budget`: size of `_selected`, the set actually replayed.
      - `pool_multiplier * quota` per class: size of `_bank`, the candidate
        pool `rebuild_selected` re-herds from. At every rebuild boundary the
        total footprint is bounded by `pool_multiplier * total_budget`.

    Mid-task the bound holds for every class that has been through a rebuild
    (its pool stays at `quota * pool_multiplier` via reservoir-bounded
    `store()`); classes awaiting their *first* rebuild deliberately keep
    their full raw stream instead — capping them early would thin the pool
    below the quota they will actually get, so the first herding would have
    no headroom. The worst-case mid-task footprint is therefore
    `pool_multiplier * total_budget` plus the full streams of at most
    `classes_per_task` classes.
    """

    def __init__(
        self,
        num_classes: int,
        total_budget: int,
        seed: int,
        floor: int = 1,
        exclude_classes: Collection[int] | None = None,
        selection: str = "herding",
        pool_multiplier: int = 3,
    ) -> None:
        if selection not in ("herding", "random"):
            raise ValueError(
                f"selection must be 'herding' or 'random', got {selection!r}"
            )
        excluded = set(exclude_classes) if exclude_classes is not None else set()
        self._bank: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self._selected: dict[int, list] = {c: [] for c in range(num_classes) if c not in excluded}
        self.class_means: dict[int, torch.Tensor] = {}
        self._seen_indices: set[int] = set()
        self._rng = random.Random(seed)
        self._total_budget = total_budget
        self._floor = floor
        self._num_classes = num_classes
        self._selection = selection

        # --- fixed-memory bookkeeping ---
        # Caps and default allocation always derive from the *live* class
        # count (``len(self._bank)``), never from the initial ``num_classes``:
        # expand() adds classes over time and the initial count goes stale.
        self._pool_multiplier = max(1, pool_multiplier)
        default_cap = max(self._floor, self._total_budget // max(1, len(self._bank))) * self._pool_multiplier
        self._pool_caps: dict[int, int] = {c: default_cap for c in self._bank}
        self._seen_count: dict[int, int] = {c: 0 for c in self._bank}
        self._quota_known: set[int] = set()

    @staticmethod
    def _to_tensor_label(y: object) -> torch.Tensor:
        if torch.is_tensor(y):
            return y
        return torch.tensor(y, dtype=torch.long)

    def start_task(self) -> None:
        self._seen_indices.clear()

    def _reservoir_insert(self, cid: int, item: tuple) -> None:
        """Bounded, class-local reservoir insertion (classic Algorithm R).

        Keeps `_bank[cid]` at or below `_pool_caps[cid]` at all times,
        including mid-task, with each item seen so far having equal
        probability of surviving in the pool. Classes that have never been
        through a `rebuild_selected` call are uncapped so they keep their
        full raw stream until a budget-derived quota is assigned.
        """
        pool = self._bank[cid]
        if cid not in self._quota_known:
            # First-ever exposure for this class: its real quota isn't known
            # yet, so don't cap it -- exactly like the original code. Capping
            # here is what silently thinned pools to floor*multiplier before
            # rebuild_selected ever got a chance to compute the real quota.
            pool.append(item)
            self._seen_count[cid] = self._seen_count.get(cid, 0) + 1
            return
        cap = self._pool_caps.get(cid, max(self._floor, 1) * self._pool_multiplier)
        self._seen_count[cid] = self._seen_count.get(cid, 0) + 1
        k = self._seen_count[cid]
        if len(pool) < cap:
            pool.append(item)
        else:
            j = self._rng.randint(1, k)
            if j <= cap:
                pool[j - 1] = item

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
            if cid in self._bank:
                self._reservoir_insert(cid, (x, y))

    def expand(self, num_new_classes: int) -> None:
        max_existing = max(self._bank.keys()) if self._bank else -1
        default_cap = max(
            self._floor,
            self._total_budget // max(1, len(self._bank) + num_new_classes),
        ) * self._pool_multiplier
        for c in range(max_existing + 1, max_existing + 1 + num_new_classes):
            if c not in self._bank:
                self._bank[c] = []
            if c not in self._selected:
                self._selected[c] = []
            self._pool_caps.setdefault(c, default_cap)
            self._seen_count.setdefault(c, 0)
        # NOTE: the default cap above is only a pre-rebuild fallback; the
        # real per-class cap (quota * pool_multiplier) is assigned inside
        # rebuild_selected once the class's budget-derived quota is known.

    @staticmethod
    def _enforce_floor(allocation: list[int], active_classes: set[int], floor: int) -> list[int]:
        """Guarantee every class with a non-empty pool keeps >= floor slots.

        Defensive, local safeguard: prevents a temporary allocation dip from
        permanently wiping a class's bank once pruning is in effect (see
        rebuild_selected). Borrows slack from whichever classes currently
        have the most. If total_budget can't cover floor * len(active_classes)
        this cannot fully succeed — that's a genuine capacity limit, not a
        bug, and should be treated as a sizing problem with total_budget/floor.
        """
        allocation = list(allocation)
        deficit = [c for c in active_classes if c < len(allocation) and allocation[c] < floor]
        if not deficit:
            return allocation
        needed = sum(floor - allocation[c] for c in deficit)
        for c in deficit:
            allocation[c] = floor
        donors = sorted(
            (c for c in range(len(allocation)) if c not in deficit),
            key=lambda c: allocation[c],
            reverse=True,
        )
        i = 0
        guard = 10_000 * max(1, len(donors))
        while needed > 0 and donors and i < guard:
            c = donors[i % len(donors)]
            if allocation[c] > floor:
                allocation[c] -= 1
                needed -= 1
            i += 1
        return allocation

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
            # they would never be re-herded.
            allocation = allocate_uniform_fixed_total(
                num_classes=len(self._bank),
                total_budget=self._total_budget,
                floor=self._floor,
            )
        active_classes = {c for c, pool in self._bank.items() if pool}
        allocation = self._enforce_floor(allocation, active_classes, self._floor)

        selected: dict[int, list] = {c: [] for c in self._bank.keys()}
        total_selected = 0
        classes_used = 0
        model.eval()
        model_device = next(model.parameters()).device
        t_rebuild_start = time.time()
        if verbose:
            print(f"[rebuild_selected] Starting. allocation len={len(allocation)}, bank keys={sorted(self._bank.keys())}", flush=True)
            for cid, pool in self._bank.items():
                print(f"  bank[{cid}] pool_size={len(pool)}", flush=True)

        with torch.inference_mode():
            for class_id, quota in enumerate(allocation):
                pool = self._bank.get(class_id, [])
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

                # --- fixed-memory pruning ---
                # Reseed the pool with this round's picks plus a bounded set
                # of leftover (non-selected) candidates, so the NEXT rebuild
                # still has real headroom to re-herd from, instead of
                # re-herding its own previous output.
                cap = max(quota, self._floor) * self._pool_multiplier
                self._pool_caps[class_id] = cap
                self._quota_known.add(class_id)
                picked_set = set(pick)
                leftover_slots = cap - len(pick)
                if leftover_slots > 0:
                    remaining_idx = [i for i in range(len(pool)) if i not in picked_set]
                    self._rng.shuffle(remaining_idx)
                    extra = [pool[i] for i in remaining_idx[:leftover_slots]]
                else:
                    extra = []
                self._bank[class_id] = selected[class_id] + extra
                self._seen_count[class_id] = len(self._bank[class_id])

        if verbose:
            print(f"[rebuild_selected] Done in {time.time()-t_rebuild_start:.2f}s. total_selected={total_selected}", flush=True)

        self._selected = selected
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

    def bank_size(self) -> int:
        """Total raw examples currently retained across all classes — the
        actual memory footprint.  Bounded by
        `pool_multiplier * total_budget` at rebuild boundaries; mid-task,
        classes that have not yet been through their first rebuild keep
        their full stream (see the class docstring)."""
        return sum(len(v) for v in self._bank.values())

    def state_dict(self) -> dict:
        return {
            "bank": {c: list(pool) for c, pool in self._bank.items()},
            "selected": {c: list(pool) for c, pool in self._selected.items()},
            "total_budget": self._total_budget,
            "floor": self._floor,
            "pool_multiplier": self._pool_multiplier,
            "pool_caps": dict(self._pool_caps),
            "seen_count": dict(self._seen_count),
            "quota_known": sorted(self._quota_known),
        }

    def load_state_dict(self, state: dict) -> None:
        self._bank = {int(c): list(pool) for c, pool in state["bank"].items()}
        self._selected = {int(c): list(pool) for c, pool in state.get("selected", {}).items()}
        self._total_budget = state.get("total_budget", self._total_budget)
        self._floor = state.get("floor", self._floor)
        self._pool_multiplier = state.get("pool_multiplier", self._pool_multiplier)
        self._pool_caps = {int(c): v for c, v in state.get("pool_caps", {}).items()}
        self._seen_count = {int(c): v for c, v in state.get("seen_count", {}).items()}
        self._quota_known = set(int(c) for c in state.get("quota_known", []))
        default_cap = max(self._floor, self._total_budget // max(1, len(self._bank))) * self._pool_multiplier
        for c in self._bank:
            self._pool_caps.setdefault(c, default_cap)
            self._seen_count.setdefault(c, len(self._bank[c]))