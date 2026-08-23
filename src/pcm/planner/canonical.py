"""Model-independent canonical Planner Cache protocol and storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pcm.planner.cache import PlannerCache, PlannerCacheConfig, SlotType


CANONICAL_VALUE_LABELS = tuple(
    "Alice Bob Clara David Elena Frank Grace Henry Irene James Karen Louis Maria Nancy "
    "Oscar Peter Queen Robert Sarah Thomas Victor Wendy Xavier London Paris Berlin Rome "
    "Cairo Tokyo Sydney garden kitchen cellar library forest castle".split()
)


def tensor_state_checksum(state: dict[str, Tensor]) -> str:
    """Deterministic checksum for portable tensor-only state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_snapshot_checksum(
    tensors: dict[str, Tensor], metadata: dict[str, str]
) -> str:
    digest = hashlib.sha256(tensor_state_checksum(tensors).encode())
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalPConfig:
    slots: int = 128
    width: int = 512
    dtype: torch.dtype = torch.float16
    device: str | torch.device = "cpu"
    merge_similarity: float = 0.92


class CanonicalPStore:
    """Fixed P slots whose serialized state has no model-hidden representation."""

    FORMAT = "pcm-canonical-p-v1"

    def __init__(self, config: CanonicalPConfig) -> None:
        self.config = config
        self.cache = PlannerCache(PlannerCacheConfig(
            slots=config.slots,
            width=config.width,
            dtype=config.dtype,
            device=config.device,
            merge_similarity=config.merge_similarity,
        ))
        device = self.cache.device
        self.entity_id = torch.full((config.slots,), -1, dtype=torch.int64, device=device)
        self.relation_id = torch.full((config.slots,), -1, dtype=torch.int64, device=device)
        self.value_id = torch.full((config.slots,), -1, dtype=torch.int64, device=device)
        self.canonical_metadata_id = torch.full(
            (config.slots,), -1, dtype=torch.int64, device=device
        )

    @property
    def canonical_values(self) -> Tensor:
        return self.cache.values

    @property
    def valid(self) -> Tensor:
        return self.cache.valid

    def create(
        self,
        canonical_value: Tensor,
        *,
        entity_id: int,
        relation_id: int,
        value_id: int,
        metadata_id: int,
        slot_type: SlotType = SlotType.FACT,
        **cache_metadata,
    ) -> tuple[int, object]:
        merge_mask = (
            (self.entity_id == entity_id)
            & (self.relation_id == relation_id)
            & self.cache.valid
        )
        slot, operation = self.cache.create(
            canonical_value, slot_type=slot_type, merge_mask=merge_mask, **cache_metadata
        )
        if slot >= 0:
            self.entity_id[slot] = entity_id
            self.relation_id[slot] = relation_id
            self.value_id[slot] = value_id
            self.canonical_metadata_id[slot] = metadata_id
        return slot, operation

    def allocation_signature(self):
        fields = (self.entity_id, self.relation_id, self.value_id, self.canonical_metadata_id)
        return self.cache.allocation_signature() + tuple(
            (field.data_ptr(), tuple(field.shape)) for field in fields
        )

    def modify(
        self,
        slot: int,
        canonical_value: Tensor,
        *,
        entity_id: int,
        relation_id: int,
        value_id: int,
        metadata_id: int,
        **cache_metadata,
    ) -> int:
        result = self.cache.modify(slot, canonical_value, **cache_metadata)
        self.entity_id[slot] = entity_id
        self.relation_id[slot] = relation_id
        self.value_id[slot] = value_id
        self.canonical_metadata_id[slot] = metadata_id
        return result

    def invalidate(self, slot: int) -> int:
        result = self.cache.invalidate(slot)
        self.entity_id[slot] = -1
        self.relation_id[slot] = -1
        self.value_id[slot] = -1
        self.canonical_metadata_id[slot] = -1
        return result

    def save(self, path: str | Path) -> None:
        tensors = {
            "canonical_values": self.cache.values.detach().cpu(),
            "valid": self.cache.valid.detach().cpu(),
            "slot_type": self.cache.slot_type.detach().cpu(),
            "confidence": self.cache.confidence.detach().cpu(),
            "importance": self.cache.importance.detach().cpu(),
            "freshness": self.cache.freshness.detach().cpu(),
            "persistence": self.cache.persistence.detach().cpu(),
            "last_updated": self.cache.last_updated.detach().cpu(),
            "source": self.cache.source.detach().cpu(),
            "entity_id": self.entity_id.detach().cpu(),
            "relation_id": self.relation_id.detach().cpu(),
            "value_id": self.value_id.detach().cpu(),
            "canonical_metadata_id": self.canonical_metadata_id.detach().cpu(),
        }
        metadata = {
            "format": self.FORMAT,
            "config": json.dumps({
                "slots": self.config.slots,
                "width": self.config.width,
                "merge_similarity": self.config.merge_similarity,
            }),
            "labels": json.dumps(self.cache.labels),
        }
        metadata["content_sha256"] = canonical_snapshot_checksum(tensors, metadata)
        save_file(tensors, str(Path(path)), metadata=metadata)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> "CanonicalPStore":
        path = Path(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if metadata.get("format") != cls.FORMAT:
            raise ValueError("unsupported canonical P snapshot")
        config = json.loads(metadata["config"])
        tensors = load_file(str(path), device=str(device))
        checksum_metadata = {
            key: value for key, value in metadata.items() if key != "content_sha256"
        }
        if canonical_snapshot_checksum(tensors, checksum_metadata) != metadata.get(
            "content_sha256"
        ):
            raise ValueError("canonical P snapshot checksum does not match")
        result = cls(CanonicalPConfig(
            slots=config["slots"], width=config["width"], dtype=dtype, device=device,
            merge_similarity=config.get("merge_similarity", 0.92),
        ))
        result.cache.values.copy_(tensors["canonical_values"].to(dtype=dtype))
        for name in (
            "valid", "slot_type", "confidence", "importance", "freshness",
            "persistence", "last_updated", "source",
        ):
            getattr(result.cache, name).copy_(tensors[name])
        for name in ("entity_id", "relation_id", "value_id", "canonical_metadata_id"):
            getattr(result, name).copy_(tensors[name])
        result.cache.labels = json.loads(metadata["labels"])
        result.cache._clock = int(result.cache.last_updated.max())
        return result


CANONICAL_P_PROTOCOL = CanonicalPStore.FORMAT


def model_config_checksum(config: object) -> str:
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    if isinstance(payload, dict):
        payload = dict(payload)
        configured_path = payload.get("_name_or_path")
        if configured_path and Path(str(configured_path)).is_absolute():
            payload["_name_or_path"] = Path(str(configured_path)).name
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
