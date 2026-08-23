"""Explicit model compatibility boundaries for canonical Planner Cache state.

TTL exposes canonical state inside a model forward path. LTL is intentionally
weaker and controls lexical output at a tokenizer or runtime boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import warnings

import torch
from safetensors import safe_open
from safetensors._safetensors_rust import SafetensorError
from safetensors.torch import load_file, save_file

from pcm.planner.canonical import CANONICAL_P_PROTOCOL
from pcm.planner.split_translator import (
    SPLIT_TRANSLATE_FORMAT,
    SplitPTranslatePackage,
    SplitTranslateConfig,
    tensor_checksum,
)


TTL_FORMAT = "planner-cache-ttl-v1"
LTL_FORMAT = "planner-cache-ltl-v1"
TTL_EXTENSION = ".ttl"
LTL_EXTENSION = ".ltl"
FORBIDDEN_ADAPTER_FIELDS = (
    "base_model",
    "conversation",
    "p_cache",
    "canonical_values",
    "optimizer",
)


class CompatibilityKind(str, Enum):
    NATIVE = "native"
    TTL = "ttl"
    LTL = "ltl"


@dataclass(frozen=True)
class CompatibilityResolution:
    kind: CompatibilityKind
    support_level: str
    artifact: Path | None


class TensorTranslationLayer(SplitPTranslatePackage):
    """Semantic/internal compatibility module backed by the Pythia TTL.

    The learned module is unchanged from the proven split translator. The new
    container identifies its stronger semantic contract and rejects LTL files.
    """

    adapter_class = CompatibilityKind.TTL.value
    support_level = "semantic/internal"
    extension = TTL_EXTENSION

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != TTL_EXTENSION:
            raise ValueError(f"TTL artifacts must use the {TTL_EXTENSION} extension")
        state = {name: value.detach().cpu() for name, value in self.state_dict().items()}
        if any(field in name.casefold() for name in state for field in FORBIDDEN_ADAPTER_FIELDS):
            raise ValueError("TTL contains forbidden model or conversation state")
        config = asdict(self.config)
        config.pop("format", None)
        manifest = {
            "format": TTL_FORMAT,
            "adapter_class": self.adapter_class,
            "support_level": self.support_level,
            "canonical_protocol": self.config.canonical_protocol,
            "config": config,
            "weights_sha256": tensor_checksum(state),
        }
        save_file(state, str(path), metadata={
            "manifest": json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        })

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        allow_legacy: bool = True,
    ) -> "TensorTranslationLayer":
        path = Path(path)
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
        except SafetensorError as error:
            raise ValueError("artifact is not a Tensor Translation Layer") from error
        manifest = json.loads(metadata["manifest"]) if "manifest" in metadata else metadata
        file_format = manifest.get("format")
        if file_format == SPLIT_TRANSLATE_FORMAT:
            if not allow_legacy:
                raise ValueError("legacy .translate artifact is not an explicit TTL")
            warnings.warn(
                ".translate is deprecated. This semantic adapter is classified as TTL.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif file_format != TTL_FORMAT:
            raise ValueError("artifact is not a Tensor Translation Layer")
        if file_format == TTL_FORMAT:
            if manifest.get("adapter_class") != CompatibilityKind.TTL.value:
                raise ValueError("TTL adapter class metadata does not match")
            if manifest.get("support_level") != "semantic/internal":
                raise ValueError("TTL support level metadata does not match")
        raw_config = manifest["config"]
        raw = json.loads(raw_config) if isinstance(raw_config, str) else dict(raw_config)
        raw["attachment_layers"] = tuple(raw["attachment_layers"])
        # The neural architecture remains the proven split translator. The
        # container format, not the in-memory architecture config, is migrated.
        raw["format"] = SPLIT_TRANSLATE_FORMAT
        result = cls(SplitTranslateConfig(**raw)).to(device=device, dtype=dtype)
        state = load_file(str(path), device=str(device))
        if any(field in name.casefold() for name in state for field in FORBIDDEN_ADAPTER_FIELDS):
            raise ValueError("TTL contains forbidden model or conversation state")
        if tensor_checksum(state) != manifest.get("weights_sha256"):
            raise ValueError("TTL weights checksum does not match")
        result.load_state_dict({name: value.to(dtype=dtype) for name, value in state.items()})
        return result


@dataclass(frozen=True)
class LexicalTranslationConfig:
    model_id: str
    model_architecture: str
    model_sha256: str
    runtime: str
    runtime_version: str
    tokenizer_bundle_sha256: str
    canonical_protocol: str = CANONICAL_P_PROTOCOL
    format: str = LTL_FORMAT
    adapter_class: str = CompatibilityKind.LTL.value
    support_level: str = "lexical/output"
    control: str = "direct_adaptive_logit_bias"
    logit_margin: float = 0.01
    parameter_count: int = 0

    def __post_init__(self) -> None:
        if self.format != LTL_FORMAT or self.adapter_class != CompatibilityKind.LTL.value:
            raise ValueError("invalid LTL format or adapter class")
        if self.support_level != "lexical/output":
            raise ValueError("invalid LTL support level")
        if self.canonical_protocol != CANONICAL_P_PROTOCOL:
            raise ValueError("unsupported canonical P protocol")
        if self.parameter_count != 0:
            raise ValueError("the direct adaptive logit-bias LTL has no learned parameters")
        if self.logit_margin < 0:
            raise ValueError("logit margin must be non-negative")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def tokenizer_bundle_checksum(path: str | Path) -> str:
    """Hash the exact tokenizer and metadata bundle used by the Gemma LTL."""
    root = Path(path)
    names = (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    )
    digest = hashlib.sha256()
    for name in names:
        item = root / name
        if not item.is_file():
            raise FileNotFoundError(f"tokenizer bundle file is missing: {item}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


class LexicalTranslationLayer:
    """Metadata-only lexical/output compatibility for llama.cpp runtimes."""

    adapter_class = CompatibilityKind.LTL.value
    support_level = "lexical/output"
    extension = LTL_EXTENSION

    def __init__(self, config: LexicalTranslationConfig) -> None:
        self.config = config

    def target(self, canonical_value: str, *, route_accepted: bool) -> str | None:
        """Return an output target only after the universal router accepts it."""
        if not route_accepted:
            return None
        value = str(canonical_value)
        return value if value else None

    def token_targets(
        self,
        canonical_value: str,
        tokenizer,
        *,
        route_accepted: bool,
    ) -> tuple[int, ...]:
        target = self.target(canonical_value, route_accepted=route_accepted)
        if target is None:
            return ()
        encoded = tokenizer(target, add_special_tokens=False).input_ids
        return tuple(int(token_id) for token_id in encoded)

    def adaptive_bias(self, logits: torch.Tensor, target_token_id: int) -> float:
        """Return the minimum non-negative bias that wins by the configured margin."""
        flat = logits.detach().float().flatten()
        if target_token_id < 0 or target_token_id >= flat.numel():
            raise IndexError("target token is outside the model vocabulary")
        masked = flat.clone()
        masked[target_token_id] = -torch.inf
        required = masked.max() - flat[target_token_id] + self.config.logit_margin
        return max(0.0, float(required))

    def validate_compatibility(
        self,
        *,
        model_id: str,
        model_architecture: str,
        model_sha256: str,
        runtime: str,
        runtime_version: str | None = None,
        tokenizer_bundle_sha256: str | None = None,
        canonical_protocol: str = CANONICAL_P_PROTOCOL,
    ) -> None:
        mismatches = []
        if model_id != self.config.model_id:
            mismatches.append("model identifier")
        if model_architecture != self.config.model_architecture:
            mismatches.append("model architecture")
        if model_sha256 != self.config.model_sha256:
            mismatches.append("model checksum")
        if runtime != self.config.runtime:
            mismatches.append("runtime")
        if runtime_version is not None and runtime_version != self.config.runtime_version:
            mismatches.append("runtime version")
        if (
            tokenizer_bundle_sha256 is not None
            and tokenizer_bundle_sha256 != self.config.tokenizer_bundle_sha256
        ):
            mismatches.append("tokenizer bundle checksum")
        if canonical_protocol != self.config.canonical_protocol:
            mismatches.append("canonical protocol")
        if mismatches:
            raise ValueError("incompatible LTL: " + ", ".join(mismatches))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix != LTL_EXTENSION:
            raise ValueError(f"LTL artifacts must use the {LTL_EXTENSION} extension")
        payload = asdict(self.config)
        payload_bytes = _canonical_json(payload)
        envelope = {
            "format": LTL_FORMAT,
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        path.write_bytes(_canonical_json(envelope) + b"\n")

    @classmethod
    def load(cls, path: str | Path) -> "LexicalTranslationLayer":
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if envelope.get("format") != LTL_FORMAT:
            raise ValueError("artifact is not a Lexical Translation Layer")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("LTL payload is missing")
        actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if actual != envelope.get("payload_sha256"):
            raise ValueError("LTL checksum does not match")
        return cls(LexicalTranslationConfig(**payload))


def classify_compatibility_artifact(path: str | Path) -> CompatibilityKind:
    """Classify modern adapters and the one supported legacy semantic format."""
    path = Path(path)
    if path.suffix == LTL_EXTENSION:
        return CompatibilityKind.LTL
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            manifest = json.loads(metadata["manifest"]) if "manifest" in metadata else metadata
            file_format = manifest.get("format")
    except Exception as error:
        raise ValueError(f"unrecognized compatibility artifact: {path}") from error
    if file_format in (TTL_FORMAT, SPLIT_TRANSLATE_FORMAT):
        return CompatibilityKind.TTL
    raise ValueError(
        "legacy artifact is research-only and has no active TTL or LTL classification"
    )


def resolve_compatibility(path: str | Path | None) -> CompatibilityResolution:
    if path is None:
        return CompatibilityResolution(CompatibilityKind.NATIVE, "native", None)
    artifact = Path(path)
    kind = classify_compatibility_artifact(artifact)
    level = "semantic/internal" if kind is CompatibilityKind.TTL else "lexical/output"
    return CompatibilityResolution(kind, level, artifact)
