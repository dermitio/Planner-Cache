"""Split model-to-canonical routing and canonical-to-model translation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from pcm.planner.canonical import CanonicalPStore
from pcm.planner.canonical import (
    CANONICAL_P_PROTOCOL,
    model_config_checksum,
    tensor_state_checksum,
)
from pcm.planner.cache import Freshness


SPLIT_TRANSLATE_FORMAT = "pcm-split-translate-v1"
CANONICAL_ROUTER_FORMAT = "pcm-canonical-router-v1"


def tensor_checksum(state: dict[str, Tensor]) -> str:
    return tensor_state_checksum(state)


class ByteEntityEncoder(nn.Module):
    """Tokenizer-independent signed byte n-gram features for open entity names."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        if width < 32:
            raise ValueError("byte entity width must be at least 32")
        self.width = width

    def encode_one(self, surface: str) -> Tensor:
        data = surface.strip().casefold().encode("utf-8")
        if not data:
            raise ValueError("entity surface form cannot be empty")
        vector = torch.zeros(self.width, dtype=torch.float32)
        for position, byte in enumerate(data):
            vector[(byte * 17 + position * 31) % self.width] += 1.0
        for position, (left, right) in enumerate(zip(data, data[1:])):
            bucket = (left * 257 + right * 17 + position * 13) % self.width
            sign = 1.0 if ((left + right + position) & 1) == 0 else -1.0
            vector[bucket] += 0.5 * sign
        return F.normalize(vector, dim=0)

    def forward(self, surfaces: list[str] | tuple[str, ...]) -> Tensor:
        return torch.stack([self.encode_one(surface) for surface in surfaces])


@dataclass
class FactorizedCanonicalQuery:
    entity: Tensor
    relation_logits: Tensor
    metadata_logits: Tensor


class ModelToCanonicalQueryProjector(nn.Module):
    def __init__(
        self,
        model_hidden_width: int,
        *,
        entity_width: int = 128,
        relation_count: int = 3,
        metadata_count: int = 4,
    ) -> None:
        super().__init__()
        self.model_hidden_width = model_hidden_width
        self.entity_width = entity_width
        self.relation_count = relation_count
        self.metadata_count = metadata_count
        self.norm = nn.LayerNorm(model_hidden_width)
        self.shared = nn.Linear(model_hidden_width, 512)
        self.entity_head = nn.Linear(512, entity_width)
        self.relation_head = nn.Linear(512, relation_count)
        self.metadata_head = nn.Linear(512, metadata_count)

    def forward(self, hidden: Tensor, entity_anchor: Tensor | None = None) -> FactorizedCanonicalQuery:
        hidden = hidden.detach().to(self.shared.weight.dtype)
        features = F.gelu(self.shared(self.norm(hidden)))
        predicted_entity = F.normalize(self.entity_head(features), dim=-1)
        if entity_anchor is not None:
            predicted_entity = F.normalize(
                entity_anchor.to(device=hidden.device, dtype=predicted_entity.dtype), dim=-1
            )
        return FactorizedCanonicalQuery(
            entity=predicted_entity,
            relation_logits=self.relation_head(features),
            metadata_logits=self.metadata_head(features),
        )


class FrozenLexicalAnchorProjector(nn.Module):
    """Experimental model-native lexical anchor mapped into canonical entity space."""

    def __init__(self, model_hidden_width: int, entity_width: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_hidden_width)
        self.projection = nn.Sequential(
            nn.Linear(model_hidden_width, 256), nn.GELU(), nn.Linear(256, entity_width)
        )

    def forward(self, lexical_hidden: Tensor) -> Tensor:
        return F.normalize(self.projection(self.norm(lexical_hidden.detach().float())), dim=-1)


@dataclass(frozen=True)
class RouterConfig:
    entity_width: int = 128
    relation_count: int = 3
    metadata_count: int = 4
    format: str = CANONICAL_ROUTER_FORMAT
    canonical_protocol: str = CANONICAL_P_PROTOCOL
    architecture: str = "canonical_factor_router_v1"


@dataclass
class CanonicalRouterIndex:
    entity: Tensor
    relation_id: Tensor
    metadata_id: Tensor
    valid: Tensor


@dataclass
class RouteResult:
    indices: Tensor
    scores: Tensor
    weights: Tensor
    features: Tensor
    accepted: Tensor
    has_valid: bool


class CanonicalPRouter(nn.Module):
    """Universal canonical-only scorer; it has no model-hidden dimensions."""

    def __init__(self, config: RouterConfig = RouterConfig()) -> None:
        super().__init__()
        self.config = config
        self.scorer = nn.Linear(4, 1)
        self.register_buffer("acceptance_threshold", torch.tensor(0.0))
        with torch.no_grad():
            self.scorer.weight.copy_(torch.tensor([[8.0, 4.0, 2.0, 2.0]]))
            self.scorer.bias.zero_()

    def build_index(
        self,
        store: CanonicalPStore,
        encoder: ByteEntityEncoder,
        *,
        device: str | torch.device,
    ) -> CanonicalRouterIndex:
        surfaces = []
        for valid, label in zip(store.valid.tolist(), store.cache.labels):
            if valid and not label:
                raise ValueError("routable canonical P slots require an entity surface label")
            surfaces.append(label if label else "<invalid>")
        entity = encoder(surfaces).to(device)
        return CanonicalRouterIndex(
            entity=entity,
            relation_id=store.relation_id.to(device),
            metadata_id=store.canonical_metadata_id.to(device),
            valid=(
                store.valid
                & (store.cache.freshness != int(Freshness.STALE))
            ).to(device),
        )

    def all_scores(
        self, query: FactorizedCanonicalQuery, index: CanonicalRouterIndex
    ) -> tuple[Tensor, Tensor]:
        entity = torch.einsum("...d,sd->...s", query.entity.float(), index.entity.float())
        relation_probability = F.softmax(query.relation_logits.float(), dim=-1)
        relation_ids = index.relation_id.clamp_min(0)
        relation = relation_probability[..., relation_ids]
        metadata_probability = F.softmax(query.metadata_logits.float(), dim=-1)
        metadata_ids = index.metadata_id.clamp_min(0)
        metadata = metadata_probability[..., metadata_ids]
        current = (index.metadata_id == 0).float().view(
            *((1,) * (entity.ndim - 1)), -1
        ).expand_as(entity)
        features = torch.stack((entity, relation, metadata, current), dim=-1)
        scores = self.scorer(features).squeeze(-1)
        valid = index.valid.view(*((1,) * (scores.ndim - 1)), -1)
        return scores.masked_fill(~valid, -torch.inf), features

    def route(
        self,
        query: FactorizedCanonicalQuery,
        index: CanonicalRouterIndex,
        *,
        top_k: int = 1,
    ) -> RouteResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores, features = self.all_scores(query, index)
        count = min(top_k, scores.shape[-1])
        if not bool(index.valid.any()):
            shape = (*scores.shape[:-1], count)
            return RouteResult(
                indices=torch.zeros(shape, dtype=torch.long, device=scores.device),
                scores=torch.full(shape, -torch.inf, device=scores.device),
                weights=torch.zeros(shape, device=scores.device),
                features=torch.zeros((*shape, 4), device=scores.device),
                accepted=torch.zeros(scores.shape[:-1], dtype=torch.bool, device=scores.device),
                has_valid=False,
            )
        selected_scores, indices = scores.topk(count, dim=-1)
        weights = torch.softmax(selected_scores, dim=-1)
        selected_features = features.gather(
            -2, indices.unsqueeze(-1).expand(*indices.shape, features.shape[-1])
        )
        return RouteResult(
            indices=indices,
            scores=selected_scores,
            weights=weights,
            features=selected_features,
            accepted=selected_scores[..., 0] >= self.acceptance_threshold,
            has_valid=bool(index.valid.any()),
        )

    def calibrate_acceptance(self, positive_scores: Tensor, negative_scores: Tensor) -> float:
        positive_scores = positive_scores.detach().float().flatten()
        negative_scores = negative_scores.detach().float().flatten()
        candidates = torch.unique(torch.cat((positive_scores, negative_scores))).sort().values
        if candidates.numel() > 1:
            candidates = (candidates[:-1] + candidates[1:]) / 2
        best_threshold = candidates[0]
        best_balanced = -1.0
        for threshold in candidates:
            true_positive = (positive_scores >= threshold).float().mean()
            true_negative = (negative_scores < threshold).float().mean()
            balanced = float((true_positive + true_negative) / 2)
            if balanced > best_balanced:
                best_balanced = balanced
                best_threshold = threshold
        self.acceptance_threshold.copy_(best_threshold.to(self.acceptance_threshold.device))
        return float(best_balanced)

    def save(self, path: str | Path) -> None:
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        save_file(state, str(path), metadata={
            "format": CANONICAL_ROUTER_FORMAT,
            "config": json.dumps(asdict(self.config), sort_keys=True),
            "weights_sha256": tensor_checksum(state),
        })

    @classmethod
    def load(cls, path: str | Path, *, device="cpu") -> "CanonicalPRouter":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if metadata.get("format") != CANONICAL_ROUTER_FORMAT:
            raise ValueError("unsupported canonical router file")
        config = RouterConfig(**json.loads(metadata["config"]))
        result = cls(config).to(device)
        state = load_file(str(path), device=str(device))
        if tensor_checksum(state) != metadata.get("weights_sha256"):
            raise ValueError("canonical router checksum does not match")
        result.load_state_dict(state)
        return result


class CanonicalValueTranslator(nn.Module):
    def __init__(self, canonical_width: int, model_hidden_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(canonical_width)
        self.input = nn.Linear(canonical_width, 512)
        self.output = nn.Linear(512, model_hidden_width)

    def forward(self, canonical: Tensor) -> Tensor:
        canonical = canonical.to(self.input.weight.dtype)
        return self.output(F.gelu(self.input(self.norm(canonical))))


class SplitInjectionGate(nn.Module):
    def __init__(self, model_hidden_width: int) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(model_hidden_width)
        self.value_norm = nn.LayerNorm(model_hidden_width)
        self.joint = nn.Linear(model_hidden_width * 2 + 4, 64)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -4.0)

    def logits(self, hidden: Tensor, translated: Tensor, route_features: Tensor) -> Tensor:
        dtype = self.joint.weight.dtype
        joint = torch.cat((
            self.hidden_norm(hidden.detach().to(dtype)),
            self.value_norm(translated.to(dtype)),
            route_features.to(dtype),
        ), dim=-1)
        return self.output(F.gelu(self.joint(joint))).squeeze(-1)

    def forward(self, hidden: Tensor, translated: Tensor, route_features: Tensor) -> Tensor:
        return torch.sigmoid(self.logits(hidden, translated, route_features))


@dataclass(frozen=True)
class SplitTranslateConfig:
    model_id: str
    model_hidden_width: int
    attachment_layers: tuple[int, ...]
    canonical_width: int = 512
    entity_width: int = 128
    relation_count: int = 3
    metadata_count: int = 4
    canonical_protocol: str = CANONICAL_P_PROTOCOL
    format: str = SPLIT_TRANSLATE_FORMAT
    architecture: str = "split_query_value_joint_gate_v1"
    model_revision: str = "local"
    model_config_sha256: str = "unspecified"
    top_k: int = 1

    def __post_init__(self):
        if self.format != SPLIT_TRANSLATE_FORMAT:
            raise ValueError("unsupported split translator format")
        if self.canonical_protocol != CANONICAL_P_PROTOCOL:
            raise ValueError("unsupported canonical P protocol")
        if self.model_hidden_width <= 0 or self.canonical_width <= 0:
            raise ValueError("translator widths must be positive")
        if not self.attachment_layers:
            raise ValueError("attachment layers cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


class SplitPTranslatePackage(nn.Module):
    """Model-specific query/value/gate modules; universal router is separate."""

    def __init__(self, config: SplitTranslateConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projector = ModelToCanonicalQueryProjector(
            config.model_hidden_width,
            entity_width=config.entity_width,
            relation_count=config.relation_count,
            metadata_count=config.metadata_count,
        )
        self.value_translator = CanonicalValueTranslator(
            config.canonical_width, config.model_hidden_width
        )
        self.gate = SplitInjectionGate(config.model_hidden_width)

    def validate_compatibility(
        self,
        *,
        model_id: str,
        model_hidden_width: int,
        canonical_protocol: str = CANONICAL_P_PROTOCOL,
        attachment_layers: tuple[int, ...] | None = None,
        model_config_sha256: str | None = None,
    ) -> None:
        errors = []
        if model_id != self.config.model_id:
            errors.append("model identifier")
        if model_hidden_width != self.config.model_hidden_width:
            errors.append("model hidden width")
        if canonical_protocol != self.config.canonical_protocol:
            errors.append("canonical protocol")
        if attachment_layers is not None and tuple(attachment_layers) != self.config.attachment_layers:
            errors.append("attachment layers")
        if (
            model_config_sha256 is not None
            and self.config.model_config_sha256 != "unspecified"
            and model_config_sha256 != self.config.model_config_sha256
        ):
            errors.append("model config checksum")
        if errors:
            raise ValueError("incompatible split translator: " + ", ".join(errors))

    def save(self, path: str | Path) -> None:
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        save_file(state, str(path), metadata={
            "format": SPLIT_TRANSLATE_FORMAT,
            "config": json.dumps(asdict(self.config), sort_keys=True),
            "weights_sha256": tensor_checksum(state),
        })

    @classmethod
    def load(cls, path: str | Path, *, device="cpu", dtype=torch.float32):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        if metadata.get("format") != SPLIT_TRANSLATE_FORMAT:
            raise ValueError("unsupported split translator file")
        raw = json.loads(metadata["config"])
        raw["attachment_layers"] = tuple(raw["attachment_layers"])
        result = cls(SplitTranslateConfig(**raw)).to(device=device, dtype=dtype)
        state = load_file(str(path), device=str(device))
        if tensor_checksum(state) != metadata.get("weights_sha256"):
            raise ValueError("split translator checksum does not match")
        result.load_state_dict({name: value.to(dtype=dtype) for name, value in state.items()})
        return result


def config_checksum(config: object) -> str:
    return model_config_checksum(config)
