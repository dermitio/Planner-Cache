import json
from pathlib import Path
import warnings
import re

import pytest
import torch

from pcm.planner.compatibility import (
    CompatibilityKind,
    LTL_FORMAT,
    TTL_FORMAT,
    LexicalTranslationConfig,
    LexicalTranslationLayer,
    TensorTranslationLayer,
    classify_compatibility_artifact,
    resolve_compatibility,
)
from pcm.planner.split_translator import SplitTranslateConfig


def tiny_ttl():
    return TensorTranslationLayer(SplitTranslateConfig(
        model_id="tiny", model_hidden_width=16, canonical_width=8,
        attachment_layers=(1,), model_config_sha256="abc",
    ))


def tiny_ltl():
    return LexicalTranslationLayer(LexicalTranslationConfig(
        model_id="gemma-test", model_architecture="gemma4",
        model_sha256="model-sha", runtime="llama.cpp",
        runtime_version="test", tokenizer_bundle_sha256="tokenizer-sha",
    ))


def test_ttl_roundtrip_is_deterministic_and_semantic(tmp_path):
    first = tmp_path / "tiny-a.ttl"
    second = tmp_path / "tiny-b.ttl"
    package = tiny_ttl()
    package.save(first)
    package.save(second)
    assert first.read_bytes() == second.read_bytes()
    restored = TensorTranslationLayer.load(first)
    assert restored.adapter_class == "ttl"
    assert restored.support_level == "semantic/internal"
    assert classify_compatibility_artifact(first) is CompatibilityKind.TTL
    for name, value in package.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


def test_ltl_roundtrip_is_deterministic_and_route_gated(tmp_path):
    first = tmp_path / "gemma-a.ltl"
    second = tmp_path / "gemma-b.ltl"
    layer = tiny_ltl()
    layer.save(first)
    layer.save(second)
    assert first.read_bytes() == second.read_bytes()
    restored = LexicalTranslationLayer.load(first)
    assert restored.config.format == LTL_FORMAT
    assert restored.target("Nathra", route_accepted=True) == "Nathra"
    assert restored.target("Nathra", route_accepted=False) is None
    assert restored.config.parameter_count == 0
    assert classify_compatibility_artifact(first) is CompatibilityKind.LTL

    class MustNotRun:
        def __call__(self, *args, **kwargs):
            raise AssertionError("tokenizer was called on an inert route")

    assert restored.token_targets(
        "Nathra", MustNotRun(), route_accepted=False,
    ) == ()
    assert restored.adaptive_bias(torch.tensor([2.0, 1.0, -1.0]), 1) == pytest.approx(1.01)


def test_ttl_and_ltl_reject_each_others_files(tmp_path):
    ttl = tmp_path / "tiny.ttl"
    ltl = tmp_path / "tiny.ltl"
    tiny_ttl().save(ttl)
    tiny_ltl().save(ltl)
    with pytest.raises(ValueError, match="not a Tensor"):
        TensorTranslationLayer.load(ltl)
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        LexicalTranslationLayer.load(ttl)


def test_ltl_checksum_model_runtime_and_protocol_validation(tmp_path):
    path = tmp_path / "gemma.ltl"
    tiny_ltl().save(path)
    envelope = json.loads(path.read_text())
    envelope["payload"]["model_id"] = "tampered"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"):
        LexicalTranslationLayer.load(path)
    layer = tiny_ltl()
    with pytest.raises(ValueError, match="model identifier"):
        layer.validate_compatibility(
            model_id="wrong", model_architecture="gemma4",
            model_sha256="model-sha", runtime="llama.cpp",
        )
    with pytest.raises(ValueError, match="runtime"):
        layer.validate_compatibility(
            model_id="gemma-test", model_architecture="gemma4",
            model_sha256="model-sha", runtime="other",
        )
    with pytest.raises(ValueError, match="runtime version"):
        layer.validate_compatibility(
            model_id="gemma-test", model_architecture="gemma4",
            model_sha256="model-sha", runtime="llama.cpp",
            runtime_version="wrong",
        )


def test_ttl_corruption_and_model_mismatch_are_rejected(tmp_path):
    path = tmp_path / "tiny.ttl"
    tiny_ttl().save(path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        TensorTranslationLayer.load(path)
    layer = tiny_ttl()
    with pytest.raises(ValueError, match="model identifier"):
        layer.validate_compatibility(model_id="wrong", model_hidden_width=16)


def test_legacy_semantic_translate_is_explicitly_deprecated(tmp_path):
    legacy = tmp_path / "legacy.translate"
    legacy_package = tiny_ttl()
    # Call the historical parent serializer to preserve its old container.
    super(TensorTranslationLayer, legacy_package).save(legacy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = TensorTranslationLayer.load(legacy)
    assert restored.adapter_class == "ttl"
    assert classify_compatibility_artifact(legacy) is CompatibilityKind.TTL
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    with pytest.raises(ValueError, match="legacy"):
        TensorTranslationLayer.load(legacy, allow_legacy=False)


def test_resolution_distinguishes_native_ttl_and_ltl(tmp_path):
    ttl = tmp_path / "tiny.ttl"
    ltl = tmp_path / "tiny.ltl"
    tiny_ttl().save(ttl)
    tiny_ltl().save(ltl)
    assert resolve_compatibility(None).kind is CompatibilityKind.NATIVE
    assert resolve_compatibility(ttl).support_level == "semantic/internal"
    assert resolve_compatibility(ltl).support_level == "lexical/output"


def test_formats_forbid_runtime_state_by_construction(tmp_path):
    ttl = tmp_path / "tiny.ttl"
    ltl = tmp_path / "tiny.ltl"
    tiny_ttl().save(ttl)
    tiny_ltl().save(ltl)
    ltl_text = ltl.read_text()
    for forbidden in ("conversation", "p_cache", "canonical_values", "base_model"):
        assert forbidden not in ltl_text
    assert TTL_FORMAT.encode() in ttl.read_bytes()


def test_session_compatibility_metadata_and_event_names(tmp_path):
    from pcm.planner.interactive_session import SessionRecorder

    recorder = SessionRecorder(
        tmp_path, "pythia", {
            "compatibility_layer": "ttl",
            "compatibility_support_level": "semantic/internal",
            "adapter_artifact": "pythia.ttl",
        },
    )
    recorder.event("TTL_ENABLE", source="ttl", gate=0.9)
    recorder.event("TTL_OUTPUT", source="ttl", active=True)
    recorder.finalize({}, reason="test")
    session = json.loads(recorder.session_path.read_text())
    events = [json.loads(row) for row in recorder.events_path.read_text().splitlines()]
    assert session["compatibility_layer"] == "ttl"
    assert [event["event"] for event in events][1:3] == ["TTL_ENABLE", "TTL_OUTPUT"]

    ltl_recorder = SessionRecorder(
        tmp_path, "gemma", {
            "compatibility_layer": "ltl",
            "compatibility_support_level": "lexical/output",
            "adapter_artifact": "gemma.ltl",
        },
    )
    ltl_recorder.event("LTL_ENABLE", source="ltl")
    ltl_recorder.event("LTL_TOKEN_TARGET", source="ltl", target_token_ids=[1, 2])
    ltl_recorder.event("LTL_COMPLETE", source="ltl")
    ltl_recorder.finalize({}, reason="test")
    ltl_events = [json.loads(row) for row in ltl_recorder.events_path.read_text().splitlines()]
    assert [event["event"] for event in ltl_events][1:4] == [
        "LTL_ENABLE", "LTL_TOKEN_TARGET", "LTL_COMPLETE",
    ]


def test_active_artifacts_have_explicit_classes_and_preserve_ttl_weights():
    root = Path(__file__).resolve().parents[1]
    legacy = root / "artifacts/pythia-1.4b-split-final_layer.translate"
    ttl_path = root / "artifacts/pythia-1.4b-final-layer.ttl"
    ltl_path = root / "artifacts/gemma4-e4b-q8-llama.ltl"
    if not all(path.is_file() for path in (legacy, ttl_path, ltl_path)):
        pytest.skip("local publication artifacts are not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_package = TensorTranslationLayer.load(legacy)
    ttl = TensorTranslationLayer.load(ttl_path)
    for name, value in legacy_package.state_dict().items():
        torch.testing.assert_close(value, ttl.state_dict()[name])
    assert LexicalTranslationLayer.load(ltl_path).config.parameter_count == 0


def test_publication_document_links_resolve():
    root = Path(__file__).resolve().parents[1]
    documents = list((root / "Publishing").rglob("*.md")) + [root / "README.md"]
    broken = []
    for document in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            local = target.split("#", 1)[0]
            if not local or "://" in local or local.startswith("mailto:"):
                continue
            if not (document.parent / local).resolve().exists():
                broken.append((str(document.relative_to(root)), target))
    assert broken == []


def test_ltl_inactive_request_is_exact_base_request():
    from pcm.planner.interactive_runtimes import gemma_chat_request_body

    messages = [{"role": "user", "content": "Where is the key?"}]
    base = gemma_chat_request_body(
        messages, max_tokens=8, temperature=0.0, top_p=1.0, seed=7,
    )
    rejected = gemma_chat_request_body(
        messages, max_tokens=8, temperature=0.0, top_p=1.0, seed=7,
        target_token_ids=(),
    )
    active = gemma_chat_request_body(
        messages, max_tokens=8, temperature=0.0, top_p=1.0, seed=7,
        target_token_ids=(17, 23),
    )
    assert rejected == base
    assert "logit_bias" not in rejected
    assert active["logit_bias"] == {"17": 100.0, "23": 100.0}


def test_ltl_request_preserves_native_structured_messages_without_aliasing():
    from pcm.planner.interactive_runtimes import gemma_chat_request_body

    messages = [
        {"role": "system", "content": "Preserve this exactly. β"},
        {"role": "user", "content": "Question with  double spaces."},
        {"role": "assistant", "content": "Prior answer\nunchanged"},
        {"role": "user", "content": "Final question?", "name": "operator"},
    ]
    request = gemma_chat_request_body(
        messages, max_tokens=8, temperature=0.0, top_p=1.0, seed=7,
    )
    assert request["messages"] == messages
    assert request["messages"] is not messages
    assert "logit_bias" not in request
    request["messages"][0]["content"] = "mutated request copy"
    assert messages[0]["content"] == "Preserve this exactly. β"
