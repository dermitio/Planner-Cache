import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.pythia_split_translate import PythiaSplitTranslatedModel
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    FactorizedCanonicalQuery,
    RouterConfig,
    SplitPTranslatePackage,
    SplitTranslateConfig,
    config_checksum,
)


def tiny_neox():
    return GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    ))


def test_byte_entity_encoder_is_deterministic_open_vocabulary():
    encoder = ByteEntityEncoder(128)
    first = encoder(["silver key", "gold key", "Silver Key"])
    second = encoder(["silver key", "gold key", "Silver Key"])
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0], first[2])
    assert float(F.cosine_similarity(first[0:1], first[1:2])) < 0.99


def test_canonical_router_separates_entity_relation_history_and_invalidation():
    encoder = ByteEntityEncoder(128)
    router = CanonicalPRouter()
    store = CanonicalPStore(CanonicalPConfig(
        slots=512, width=16, dtype=torch.float32, merge_similarity=1.0
    ))
    correct, _ = store.create(
        torch.randn(16), entity_id=0, relation_id=0, value_id=0, metadata_id=0,
        label="silver key",
    )
    store.create(
        torch.randn(16), entity_id=1, relation_id=0, value_id=0, metadata_id=0,
        label="gold key",
    )
    store.create(
        torch.randn(16), entity_id=0, relation_id=1, value_id=0, metadata_id=0,
        label="silver key",
    )
    historical, _ = store.create(
        torch.randn(16), entity_id=0, relation_id=0, value_id=0, metadata_id=2,
        label="silver key",
    )
    for index in range(4, 512):
        store.create(
            torch.randn(16), entity_id=index, relation_id=index % 3,
            value_id=0, metadata_id=0, label=f"irrelevant entity {index}",
        )
    query = FactorizedCanonicalQuery(
        entity=encoder(["silver key"]),
        relation_logits=torch.tensor([[8.0, -8.0, -8.0]]),
        metadata_logits=torch.tensor([[8.0, -8.0, -8.0, -8.0]]),
    )
    index = router.build_index(store, encoder, device="cpu")
    route = router.route(query, index, top_k=4)
    assert int(route.indices[0, 0]) == correct
    historical_position = route.indices[0].tolist().index(historical)
    assert float(route.scores[0, 0].detach()) > float(
        route.scores[0, historical_position].detach()
    )
    store.invalidate(correct)
    index = router.build_index(store, encoder, device="cpu")
    scores, _ = router.all_scores(query, index)
    assert torch.isneginf(scores[0, correct])


def test_split_package_and_router_roundtrip_without_base_or_p_state(tmp_path):
    base = tiny_neox()
    config = SplitTranslateConfig(
        model_id="gpt_neox", model_hidden_width=32, canonical_width=16,
        attachment_layers=(1,), model_config_sha256=config_checksum(base.config),
    )
    package = SplitPTranslatePackage(config)
    router = CanonicalPRouter(RouterConfig())
    package_path = tmp_path / "tiny.translate"
    router_path = tmp_path / "canonical.router"
    package.save(package_path)
    router.save(router_path)
    restored = SplitPTranslatePackage.load(package_path)
    restored_router = CanonicalPRouter.load(router_path)
    assert restored.config == package.config
    assert restored_router.config == router.config
    for path in (package_path, router_path):
        tensors = load_file(path)
        assert not any(
            forbidden in name
            for name in tensors
            for forbidden in ("base_model", "canonical_values", "conversation", "p_cache")
        )


def test_tiny_pythia_split_path_freezes_base_and_invalidation_is_identity():
    base = tiny_neox().eval()
    package = SplitPTranslatePackage(SplitTranslateConfig(
        model_id="gpt_neox", model_hidden_width=32, canonical_width=16,
        attachment_layers=(1,), model_config_sha256=config_checksum(base.config),
    ))
    wrapper = PythiaSplitTranslatedModel(
        base, package, CanonicalPRouter(), ByteEntityEncoder()
    ).train()
    store = CanonicalPStore(CanonicalPConfig(
        slots=4, width=16, dtype=torch.float32, merge_similarity=1.0
    ))
    slot, _ = store.create(
        torch.randn(16), entity_id=0, relation_id=0, value_id=0, metadata_id=0,
        label="silver key",
    )
    tokens = torch.randint(0, 100, (1, 8))
    disabled = wrapper(input_ids=tokens, use_cache=False).logits.detach()
    enabled = wrapper(input_ids=tokens, p_store=store, use_cache=False).logits
    instrumented = wrapper(
        input_ids=tokens, p_store=store, collect_telemetry=True, use_cache=False
    ).logits
    torch.testing.assert_close(instrumented, enabled)
    assert wrapper.query_telemetry
    assert wrapper.route_telemetry
    assert wrapper.gate_telemetry
    enabled.square().mean().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in package.parameters())
    store.invalidate(slot)
    invalidated = wrapper(input_ids=tokens, p_store=store, use_cache=False).logits
    torch.testing.assert_close(invalidated, disabled)
    wrapper.close()
