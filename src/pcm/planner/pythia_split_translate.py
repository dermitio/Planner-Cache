"""Frozen-Pythia attachment for the split canonical router/translator."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from pathlib import Path

from pcm.planner.canonical import CanonicalPStore
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    CanonicalRouterIndex,
    RouteResult,
    FactorizedCanonicalQuery,
    SplitPTranslatePackage,
    config_checksum,
)


def pythia_model_identifier(base_model: nn.Module) -> str:
    configured = str(getattr(base_model.config, "_name_or_path", "")).strip()
    if configured and Path(configured).is_absolute():
        configured = Path(configured).name
    return configured or str(getattr(base_model.config, "model_type", "gpt_neox"))


class PythiaSplitTranslatedModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        package: SplitPTranslatePackage,
        router: CanonicalPRouter,
        entity_encoder: ByteEntityEncoder,
    ) -> None:
        super().__init__()
        if not hasattr(base_model, "gpt_neox"):
            raise TypeError("base model must expose GPT-NeoX transformer layers")
        package.validate_compatibility(
            model_id=pythia_model_identifier(base_model),
            model_hidden_width=int(base_model.config.hidden_size),
            attachment_layers=package.config.attachment_layers,
            model_config_sha256=config_checksum(base_model.config),
        )
        self.base_model = base_model
        self.package = package
        self.router = router
        self.entity_encoder = entity_encoder
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        layers = base_model.gpt_neox.layers
        if any(index < 0 or index >= len(layers) for index in package.config.attachment_layers):
            raise IndexError("split translator attachment layer is outside Pythia depth")
        self._store: CanonicalPStore | None = None
        self._index: CanonicalRouterIndex | None = None
        self._oracle_indices: Tensor | None = None
        self._query_entity_anchor: Tensor | None = None
        self._gate_enabled = True
        self._injection_enabled = True
        self._collect = False
        self._gate_telemetry: list[Tensor] = []
        self._route_telemetry: list[RouteResult] = []
        self._query_telemetry: list[FactorizedCanonicalQuery] = []
        self._handles = [
            layers[index].register_forward_hook(self._hook)
            for index in package.config.attachment_layers
        ]
        self.base_model.eval()

    def _oracle_route(self, query, hidden: Tensor) -> RouteResult:
        assert self._index is not None and self._oracle_indices is not None
        scores, features = self.router.all_scores(query, self._index)
        batch, sequence = hidden.shape[:2]
        indices = self._oracle_indices.to(hidden.device).view(batch, 1, 1).expand(batch, sequence, 1)
        selected_scores = scores.gather(-1, indices)
        selected_features = features.gather(
            -2, indices.unsqueeze(-1).expand(batch, sequence, 1, 4)
        )
        return RouteResult(
            indices=indices,
            scores=selected_scores,
            weights=torch.ones_like(selected_scores),
            features=selected_features,
            accepted=torch.ones((batch, sequence), dtype=torch.bool, device=hidden.device),
            has_valid=True,
        )

    def _hook(self, _module, _inputs, hidden: Tensor):
        if self._store is None or self._store.cache.occupied == 0:
            return hidden
        assert self._index is not None
        entity_anchor = None
        if self._query_entity_anchor is not None:
            entity_anchor = self._query_entity_anchor[:, None, :].expand(
                hidden.shape[0], hidden.shape[1], -1
            )
        query = self.package.query_projector(hidden, entity_anchor=entity_anchor)
        if self._collect:
            self._query_telemetry.append(FactorizedCanonicalQuery(
                entity=query.entity.detach(),
                relation_logits=query.relation_logits.detach(),
                metadata_logits=query.metadata_logits.detach(),
            ))
        route = (
            self._oracle_route(query, hidden)
            if self._oracle_indices is not None
            else self.router.route(query, self._index, top_k=self.package.config.top_k)
        )
        if self._collect:
            self._route_telemetry.append(RouteResult(
                indices=route.indices.detach(), scores=route.scores.detach(),
                weights=route.weights.detach(), features=route.features.detach(),
                accepted=route.accepted.detach(),
                has_valid=route.has_valid,
            ))
        if not self._injection_enabled or not route.has_valid:
            return hidden
        canonical = self._store.canonical_values.to(
            device=hidden.device, dtype=route.weights.dtype
        )
        selected = canonical[route.indices]
        pooled = torch.einsum("...k,...kd->...d", route.weights, selected)
        translated = self.package.value_translator(pooled)
        route_features = torch.einsum(
            "...k,...kf->...f", route.weights, route.features
        )
        gate = (
            self.package.gate(hidden, translated, route_features)
            if self._gate_enabled
            else torch.ones(hidden.shape[:-1], device=hidden.device, dtype=translated.dtype)
        )
        gate = gate * route.accepted.to(gate.dtype)
        if self._collect:
            self._gate_telemetry.append(gate.detach())
        return hidden + (gate.unsqueeze(-1) * translated).to(hidden.dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.package.train(mode)
        self.router.train(mode)
        return self

    def forward(
        self,
        *args,
        p_store: CanonicalPStore | None = None,
        query_entity_surfaces: list[str] | tuple[str, ...] | None = None,
        oracle_indices: Tensor | None = None,
        gate_enabled: bool = True,
        injection_enabled: bool = True,
        collect_telemetry: bool = False,
        **kwargs,
    ):
        if self._store is not None:
            raise RuntimeError("PythiaSplitTranslatedModel is not reentrant")
        self._store = p_store
        self._oracle_indices = oracle_indices
        if query_entity_surfaces is not None:
            if "input_ids" in kwargs and len(query_entity_surfaces) != kwargs["input_ids"].shape[0]:
                raise ValueError("query entity surface count must match the input batch")
            self._query_entity_anchor = self.entity_encoder(
                list(query_entity_surfaces)
            ).to(next(self.package.parameters()).device)
        else:
            self._query_entity_anchor = None
        self._gate_enabled = gate_enabled
        self._injection_enabled = injection_enabled
        self._collect = collect_telemetry
        self._gate_telemetry.clear()
        self._route_telemetry.clear()
        self._query_telemetry.clear()
        if p_store is not None and p_store.cache.occupied:
            self._index = self.router.build_index(
                p_store, self.entity_encoder, device=next(self.package.parameters()).device
            )
        try:
            return self.base_model(*args, **kwargs)
        finally:
            self._store = None
            self._index = None
            self._oracle_indices = None
            self._query_entity_anchor = None
            self._collect = False

    @property
    def gate_telemetry(self):
        return tuple(self._gate_telemetry)

    @property
    def route_telemetry(self):
        return tuple(self._route_telemetry)

    @property
    def query_telemetry(self):
        return tuple(self._query_telemetry)

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
