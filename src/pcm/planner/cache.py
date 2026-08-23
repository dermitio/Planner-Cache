"""Fixed-allocation first-class planner state cache."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F


class StateOperation(IntEnum):
    KEEP = 0
    CREATE = 1
    MODIFY = 2
    MERGE = 3
    INVALIDATE = 4
    IGNORE = 5


class SlotType(IntEnum):
    GOAL = 0
    ENTITY = 1
    FACT = 2
    HYPOTHESIS = 3
    CONSTRAINT = 4
    TASK = 5
    LATENT = 6
    EXTERNAL = 7


class Freshness(IntEnum):
    FRESH = 0
    STALE = 1
    UNKNOWN = 2


class Persistence(IntEnum):
    PERMANENT = 0
    DURABLE = 1
    SESSION = 2
    EXTERNAL = 3
    VOLATILE = 4


class SlotSource(IntEnum):
    CONVERSATION = 0
    RETRIEVAL = 1
    CORRECTION = 2
    TOOL = 3
    INFERENCE = 4


class CacheFullProtectedError(RuntimeError):
    """Raised when every physical slot is occupied by permanent state."""


@dataclass(frozen=True)
class PlannerCacheConfig:
    slots: int = 128
    width: int = 512
    dtype: torch.dtype = torch.float16
    device: str | torch.device = "cpu"
    merge_similarity: float = 0.92

    def __post_init__(self) -> None:
        if self.slots <= 0 or self.width <= 0:
            raise ValueError("planner slots and width must be positive")
        if not -1.0 <= self.merge_similarity <= 1.0:
            raise ValueError("merge_similarity must be between -1 and 1")


class PlannerCache:
    """Preallocated planner values and metadata mutated strictly in place."""

    def __init__(self, config: PlannerCacheConfig) -> None:
        self.config = config
        device = torch.device(config.device)
        self.values = torch.zeros((config.slots, config.width), dtype=config.dtype, device=device)
        self.valid = torch.zeros(config.slots, dtype=torch.bool, device=device)
        self.slot_type = torch.full((config.slots,), int(SlotType.LATENT), dtype=torch.int8, device=device)
        self.confidence = torch.zeros(config.slots, dtype=torch.float32, device=device)
        self.importance = torch.zeros(config.slots, dtype=torch.float32, device=device)
        self.freshness = torch.full((config.slots,), int(Freshness.UNKNOWN), dtype=torch.int8, device=device)
        self.persistence = torch.full((config.slots,), int(Persistence.VOLATILE), dtype=torch.int8, device=device)
        self.last_updated = torch.zeros(config.slots, dtype=torch.int64, device=device)
        self.source = torch.full((config.slots,), int(SlotSource.INFERENCE), dtype=torch.int8, device=device)
        self.labels: list[str | None] = [None] * config.slots
        self._clock = 0

    @property
    def device(self) -> torch.device:
        return self.values.device

    def allocation_signature(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Stable identity/shape signature for physical-allocation tests."""
        tensors = (
            self.values,
            self.valid,
            self.slot_type,
            self.confidence,
            self.importance,
            self.freshness,
            self.persistence,
            self.last_updated,
            self.source,
        )
        return tuple((tensor.data_ptr(), tuple(tensor.shape)) for tensor in tensors)

    @property
    def occupied(self) -> int:
        return int(self.valid.sum().item())

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _value(self, value: Tensor) -> Tensor:
        value = value.detach().to(device=self.device, dtype=self.config.dtype)
        if value.shape != (self.config.width,):
            raise ValueError(f"planner value must have shape ({self.config.width},)")
        return value

    def _require_valid(self, index: int) -> None:
        if not 0 <= index < self.config.slots or not bool(self.valid[index]):
            raise IndexError(f"planner slot {index} is not valid")

    def _write_metadata(
        self,
        index: int,
        *,
        slot_type: SlotType,
        confidence: float,
        importance: float,
        freshness: Freshness,
        persistence: Persistence,
        source: SlotSource,
        label: str | None,
    ) -> None:
        self._validate_score("confidence", confidence)
        self._validate_score("importance", importance)
        slot_type = SlotType(slot_type)
        freshness = Freshness(freshness)
        persistence = Persistence(persistence)
        source = SlotSource(source)
        if label is not None and not isinstance(label, str):
            raise TypeError("planner label must be a string or None")
        self.slot_type[index] = int(slot_type)
        self.confidence[index] = confidence
        self.importance[index] = importance
        self.freshness[index] = int(freshness)
        self.persistence[index] = int(persistence)
        self.source[index] = int(source)
        self.last_updated[index] = self._tick()
        self.labels[index] = label
        self.valid[index] = True

    @staticmethod
    def _validate_score(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number in [0, 1]")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    def _merge_candidate(
        self, value: Tensor, slot_type: SlotType, merge_mask: Tensor | None = None
    ) -> int | None:
        compatible = self.valid & (self.slot_type == int(slot_type))
        if merge_mask is not None:
            merge_mask = merge_mask.detach().to(device=self.device, dtype=torch.bool)
            if merge_mask.shape != self.valid.shape:
                raise ValueError("merge mask must match the planner slot shape")
            compatible &= merge_mask
        indices = compatible.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return None
        candidates = self.values.index_select(0, indices).float()
        similarities = F.cosine_similarity(candidates, value.float().unsqueeze(0), dim=-1)
        best = int(similarities.argmax().item())
        if float(similarities[best]) < self.config.merge_similarity:
            return None
        return int(indices[best].item())

    def _eviction_candidate(self) -> int:
        candidates = self.valid & (self.persistence != int(Persistence.PERMANENT))
        indices = candidates.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise CacheFullProtectedError("all planner slots are permanent")
        age = (self._clock + 1 - self.last_updated.index_select(0, indices)).float()
        stale_bonus = (self.freshness.index_select(0, indices) != int(Freshness.FRESH)).float()
        persistence_cost = torch.tensor(
            [4.0, 3.0, 2.0, 1.0, 0.0], device=self.device
        ).index_select(0, self.persistence.index_select(0, indices).long())
        keep_score = (
            4.0 * self.importance.index_select(0, indices)
            + self.confidence.index_select(0, indices)
            + persistence_cost
            - stale_bonus
            - age * 1e-6
        )
        return int(indices[int(keep_score.argmin().item())].item())

    @staticmethod
    def _admission_score(
        *,
        importance: float,
        confidence: float,
        freshness: Freshness,
        persistence: Persistence,
    ) -> float:
        persistence_cost = (4.0, 3.0, 2.0, 1.0, 0.0)[int(persistence)]
        stale_cost = 0.0 if freshness == Freshness.FRESH else 1.0
        return 4.0 * importance + confidence + persistence_cost - stale_cost

    def _slot_admission_score(self, index: int) -> float:
        age = (self._clock + 1 - int(self.last_updated[index])) * 1e-6
        return self._admission_score(
            importance=float(self.importance[index]),
            confidence=float(self.confidence[index]),
            freshness=Freshness(int(self.freshness[index])),
            persistence=Persistence(int(self.persistence[index])),
        ) - age

    def create(
        self,
        value: Tensor,
        *,
        slot_type: SlotType = SlotType.LATENT,
        confidence: float = 1.0,
        importance: float = 0.5,
        freshness: Freshness = Freshness.FRESH,
        persistence: Persistence = Persistence.SESSION,
        source: SlotSource = SlotSource.CONVERSATION,
        label: str | None = None,
        merge_mask: Tensor | None = None,
    ) -> tuple[int, StateOperation]:
        value = self._value(value)
        self._validate_score("confidence", confidence)
        self._validate_score("importance", importance)
        slot_type = SlotType(slot_type)
        freshness = Freshness(freshness)
        persistence = Persistence(persistence)
        source = SlotSource(source)
        merge_index = self._merge_candidate(value, slot_type, merge_mask)
        if merge_index is not None:
            self.merge((merge_index,), value=value, confidence=confidence, source=source)
            self.importance[merge_index] = max(
                float(self.importance[merge_index]), importance
            )
            self.persistence[merge_index] = min(
                int(self.persistence[merge_index]), int(persistence)
            )
            if label is not None:
                self.labels[merge_index] = label
            return merge_index, StateOperation.MERGE
        free = (~self.valid).nonzero(as_tuple=False).flatten()
        operation = StateOperation.CREATE
        if free.numel():
            index = int(free[0].item())
        else:
            index = self._eviction_candidate()
            incoming_score = self._admission_score(
                importance=importance,
                confidence=confidence,
                freshness=freshness,
                persistence=persistence,
            )
            if incoming_score <= self._slot_admission_score(index):
                return -1, StateOperation.IGNORE
            self.invalidate(index)
        self.values[index].copy_(value)
        self._write_metadata(
            index,
            slot_type=slot_type,
            confidence=confidence,
            importance=importance,
            freshness=freshness,
            persistence=persistence,
            source=source,
            label=label,
        )
        return index, operation

    def keep(self, index: int, *, confidence: float | None = None) -> int:
        self._require_valid(index)
        if confidence is not None:
            self._validate_score("confidence", confidence)
            self.confidence[index] = confidence
        self.last_updated[index] = self._tick()
        return index

    def modify(
        self,
        index: int,
        value: Tensor,
        *,
        confidence: float | None = None,
        freshness: Freshness = Freshness.FRESH,
        source: SlotSource | None = None,
    ) -> int:
        self._require_valid(index)
        freshness = Freshness(freshness)
        if source is not None:
            source = SlotSource(source)
        if confidence is not None:
            self._validate_score("confidence", confidence)
        # A model inference is lower-authority than an explicit user
        # correction and cannot silently overwrite it.
        if (
            source == SlotSource.INFERENCE
            and int(self.source[index]) == int(SlotSource.CORRECTION)
        ):
            self.last_updated[index] = self._tick()
            return index
        self.values[index].copy_(self._value(value))
        if confidence is not None:
            self.confidence[index] = confidence
        self.freshness[index] = int(freshness)
        if source is not None:
            self.source[index] = int(source)
        self.last_updated[index] = self._tick()
        return index

    def merge(
        self,
        indices: Iterable[int],
        *,
        value: Tensor | None = None,
        confidence: float | None = None,
        source: SlotSource = SlotSource.INFERENCE,
    ) -> int:
        indices = tuple(dict.fromkeys(indices))
        if not indices:
            raise ValueError("merge requires at least one slot")
        for index in indices:
            self._require_valid(index)
        target = max(indices, key=lambda index: float(self.importance[index]))
        merged = self._value(value) if value is not None else self.values[list(indices)].float().mean(0).to(self.config.dtype)
        self.values[target].copy_(merged)
        if confidence is None:
            confidence = max(float(self.confidence[index]) for index in indices)
        self._validate_score("confidence", confidence)
        self.confidence[target] = confidence
        self.importance[target] = max(float(self.importance[index]) for index in indices)
        self.freshness[target] = int(Freshness.FRESH)
        self.source[target] = int(source)
        self.last_updated[target] = self._tick()
        for index in indices:
            if index != target:
                self.invalidate(index)
        return target

    def invalidate(self, index: int) -> int:
        self._require_valid(index)
        self.valid[index] = False
        self.values[index].zero_()
        self.labels[index] = None
        self.last_updated[index] = self._tick()
        return index

    def apply(self, operation: StateOperation, **kwargs):
        if operation == StateOperation.KEEP:
            return self.keep(**kwargs)
        if operation == StateOperation.CREATE:
            return self.create(**kwargs)
        if operation == StateOperation.MODIFY:
            return self.modify(**kwargs)
        if operation == StateOperation.MERGE:
            return self.merge(**kwargs)
        if operation == StateOperation.INVALIDATE:
            return self.invalidate(**kwargs)
        if operation == StateOperation.IGNORE:
            return None
        raise ValueError(f"unsupported planner operation: {operation}")
