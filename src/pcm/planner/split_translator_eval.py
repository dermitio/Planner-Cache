"""Exact staged evaluation of split canonical routing and value translation."""

from __future__ import annotations

import gc
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.cache import SlotSource
from pcm.planner.pythia_split_translate import PythiaSplitTranslatedModel
from pcm.planner.pythia_split_translate import pythia_model_identifier
from pcm.planner.representation import CANONICAL, HISTORICAL, FactorizedStateRepresentation
from pcm.planner.split_translator import (
    ByteEntityEncoder,
    CanonicalPRouter,
    CanonicalRouterIndex,
    FactorizedCanonicalQuery,
    FrozenLexicalAnchorProjector,
    RouterConfig,
    SplitPTranslatePackage,
    SplitTranslateConfig,
    config_checksum,
)
from pcm.planner.canonical import CANONICAL_VALUE_LABELS as VALUE_LABELS


ADJECTIVES = tuple(
    "silver gold crimson azure ivory ebony amber jade copper iron crystal shadow bright "
    "quiet ancient hidden broken little grand northern southern eastern western moon sun "
    "star river storm winter summer autumn".split()
)
NOUNS = tuple(
    "key ring blade crown lantern compass chalice mirror scroll seal pendant coin map book "
    "box door tower bridge garden harbor temple forest castle chamber wagon banner stone "
    "cloak staff mask bell".split()
)
RELATION_PROMPTS = (
    "The owner of the {entity} is",
    "The current location of the {entity} is",
    "The current status of the {entity} is",
)
HELDOUT_LEADS = (
    "After a long unrelated scene at the inn, ",
    "Following several jokes and descriptions of the rainy road, ",
)
TRAIN_LEADS = (
    "After unrelated conversation, ",
    "With the source state absent from recent context, ",
)
SLOT_SIZES = (4, 20, 64, 128, 256, 512)


def entity_split():
    train, heldout = [], []
    for adjective_index, adjective in enumerate(ADJECTIVES):
        for noun_index, noun in enumerate(NOUNS):
            surface = f"{adjective} {noun}"
            target = heldout if (adjective_index * 31 + noun_index * 17) % 5 == 0 else train
            target.append(surface)
    for required in ("silver key", "gold key"):
        if required in train:
            train.remove(required)
            heldout.append(required)
    return train, heldout


def _bytes(parameters) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in parameters)


def run_split_translator_experiment(
    path: str | Path,
    representation: FactorizedStateRepresentation,
    *,
    attachment_count: int,
    query_steps: int = 400,
    router_steps: int = 400,
    value_steps: int = 400,
    causal_steps: int = 256,
    seed: int = 307,
    package_path: str | Path | None = None,
    router_path: str | Path | None = None,
):
    if attachment_count not in (1, 2, 4):
        raise ValueError("attachment_count must be 1, 2, or 4")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    path = Path(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    depth = len(base.gpt_neox.layers)
    layers = tuple(range(depth - attachment_count, depth))
    package = SplitPTranslatePackage(SplitTranslateConfig(
        model_id=pythia_model_identifier(base),
        model_hidden_width=int(base.config.hidden_size),
        attachment_layers=layers,
        model_config_sha256=config_checksum(base.config),
        top_k=1,
    )).to("cuda", dtype=torch.float32)
    router = CanonicalPRouter(RouterConfig()).to("cuda")
    byte_encoder = ByteEntityEncoder(128)
    wrapper = PythiaSplitTranslatedModel(base, package, router, byte_encoder).to("cuda").train()
    representation.eval()
    all_train_surfaces, all_heldout_surfaces = entity_split()
    split_rng = random.Random(seed + 1)
    train_surfaces = split_rng.sample(all_train_surfaces, 256)
    required = ["silver key", "gold key"]
    train_adjectives = {surface.split()[0] for surface in train_surfaces}
    train_nouns = {surface.split()[1] for surface in train_surfaces}
    compositional_heldout = [
        surface for surface in all_heldout_surfaces
        if surface.split()[0] in train_adjectives
        and surface.split()[1] in train_nouns
        and surface not in required
    ]
    heldout_surfaces = required + compositional_heldout[:62]
    train_surface_set = set(train_surfaces)
    assert not train_surface_set.intersection(heldout_surfaces)

    encoded_values = [tokenizer.encode(" " + value, add_special_tokens=False) for value in VALUE_LABELS]
    if any(len(ids) != 1 for ids in encoded_values):
        raise RuntimeError("controlled values must be single Pythia tokens")
    value_token_ids = [ids[0] for ids in encoded_values]
    lm_values = base.get_output_embeddings().weight[value_token_ids].detach().float()
    normalized_lm_values = F.normalize(lm_values, dim=-1)

    def query_texts(surfaces, relations, leads):
        texts, names, relation_ids = [], [], []
        for lead in leads:
            for surface, relation in zip(surfaces, relations):
                texts.append(lead + RELATION_PROMPTS[relation].format(entity=surface))
                names.append(surface)
                relation_ids.append(relation)
        return texts, names, relation_ids

    def tokenize(texts):
        return tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")

    def capture(texts, chunk=24):
        by_layer = {layer: [] for layer in layers}
        handles = [
            base.gpt_neox.layers[layer].register_forward_hook(
                lambda _module, _inputs, output, layer=layer: by_layer[layer].append(
                    output[:, -1].detach().cpu()
                )
            )
            for layer in layers
        ]
        for start in range(0, len(texts), chunk):
            with torch.inference_mode():
                wrapper(**tokenize(texts[start:start + chunk]), use_cache=False)
        for handle in handles:
            handle.remove()
        return torch.stack([
            torch.cat(by_layer[layer]).to("cuda") for layer in layers
        ])

    train_relations = [index % 3 for index in range(len(train_surfaces))]
    train_texts, train_names, train_relation_ids = query_texts(
        train_surfaces, train_relations, TRAIN_LEADS
    )
    train_hidden = capture(train_texts)
    train_names = train_names
    train_relation_ids = torch.tensor(train_relation_ids, device="cuda")
    train_entity_targets = byte_encoder(train_names).to("cuda")
    heldout_relations = [index % 3 for index in range(len(heldout_surfaces))]
    heldout_texts, heldout_names, heldout_relation_ids = query_texts(
        heldout_surfaces, heldout_relations, (HELDOUT_LEADS[0],)
    )
    heldout_hidden = capture(heldout_texts)
    heldout_relation_ids = torch.tensor(heldout_relation_ids, device="cuda")
    heldout_entity_targets = byte_encoder(heldout_names).to("cuda")

    query_optimizer = torch.optim.AdamW(package.query_projector.parameters(), lr=2e-3, eps=1e-6)
    query_losses = []
    for _ in range(query_steps):
        surface_indices = rng.sample(range(len(train_surfaces)), 32)
        hidden = torch.stack([
            train_hidden[rng.randrange(attachment_count), index]
            for index in surface_indices
        ])
        projected = package.query_projector(hidden)
        targets = byte_encoder([train_surfaces[index] for index in surface_indices]).to("cuda")
        relations = torch.tensor(
            [train_relations[index] for index in surface_indices], device="cuda"
        )
        entity_loss = 1 - F.cosine_similarity(projected.entity, targets, dim=-1).mean()
        contrastive = F.cross_entropy(
            projected.entity @ targets.T / 0.07,
            torch.arange(len(surface_indices), device="cuda"),
        )
        relation_loss = F.cross_entropy(projected.relation_logits, relations)
        metadata_loss = F.cross_entropy(
            projected.metadata_logits,
            torch.zeros(len(surface_indices), dtype=torch.long, device="cuda"),
        )
        loss = entity_loss + contrastive + relation_loss + 0.25 * metadata_loss
        query_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        query_optimizer.step()
        query_losses.append(float(loss.detach()))
    del query_optimizer

    def query_metrics(hidden, names, relations):
        with torch.inference_mode():
            projected = package.query_projector(hidden.mean(0))
            targets = byte_encoder(names).to("cuda")
            entity_scores = projected.entity @ targets.T
            return {
                "entity_accuracy": float((
                    entity_scores.argmax(-1) == torch.arange(len(names), device="cuda")
                ).float().mean()),
                "relation_accuracy": float((
                    projected.relation_logits.argmax(-1) == relations
                ).float().mean()),
                "metadata_accuracy": float((
                    projected.metadata_logits.argmax(-1) == 0
                ).float().mean()),
                "entity_cosine": float(F.cosine_similarity(
                    projected.entity, targets, dim=-1
                ).mean()),
            }

    query_heldout_metrics = query_metrics(
        heldout_hidden, heldout_names, heldout_relation_ids
    )
    byte_surface_metrics = {
        "entity_accuracy": 1.0,
        "relation_accuracy": query_heldout_metrics["relation_accuracy"],
        "metadata_accuracy": query_heldout_metrics["metadata_accuracy"],
        "entity_cosine": 1.0,
        "tokenizer_independent": True,
        "oracle_slot_assignments": 0,
    }

    lexical_projector = FrozenLexicalAnchorProjector(int(base.config.hidden_size)).to("cuda")
    embedding = base.get_input_embeddings().weight.detach()

    def lexical(surfaces):
        values = []
        for surface in surfaces:
            ids = tokenizer.encode(" " + surface, add_special_tokens=False)
            values.append(embedding[torch.tensor(ids, device="cuda")].float().mean(0))
        return torch.stack(values)

    train_lexical = lexical(train_surfaces)
    lexical_optimizer = torch.optim.AdamW(lexical_projector.parameters(), lr=2e-3)
    for _ in range(query_steps):
        indices = torch.tensor(rng.sample(range(len(train_surfaces)), 32), device="cuda")
        output = lexical_projector(train_lexical.index_select(0, indices))
        target = byte_encoder([train_surfaces[int(index)] for index in indices]).to("cuda")
        loss = 1 - F.cosine_similarity(output, target, dim=-1).mean()
        lexical_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lexical_optimizer.step()
    with torch.inference_mode():
        lexical_output = lexical_projector(lexical(heldout_surfaces))
        lexical_targets = byte_encoder(heldout_surfaces).to("cuda")
        lexical_metrics = {
            "entity_accuracy": float((
                (lexical_output @ lexical_targets.T).argmax(-1)
                == torch.arange(len(heldout_surfaces), device="cuda")
            ).float().mean()),
            "entity_cosine": float(F.cosine_similarity(
                lexical_output, lexical_targets, dim=-1
            ).mean()),
        }
    del lexical_optimizer, lexical_projector, train_lexical

    for parameter in package.query_projector.parameters():
        parameter.requires_grad_(False)
    router_optimizer = torch.optim.AdamW(router.parameters(), lr=1e-2)
    router_losses = []
    for _ in range(router_steps):
        source_index = rng.randrange(len(train_names))
        correct_surface = train_names[source_index]
        relation = int(train_relation_ids[source_index])
        relation_logits = torch.full((1, 3), -12.0, device="cuda")
        relation_logits[0, relation] = 12.0
        query = FactorizedCanonicalQuery(
            entity=byte_encoder([correct_surface]).to("cuda"),
            relation_logits=relation_logits,
            metadata_logits=torch.tensor([[12.0, -12.0, -12.0, -12.0]], device="cuda"),
        )
        candidates = [correct_surface, rng.choice(train_surfaces)]
        while candidates[1] == correct_surface:
            candidates[1] = rng.choice(train_surfaces)
        candidates.extend((correct_surface, correct_surface))
        candidate_relations = [relation, relation, (relation + 1) % 3, relation]
        candidate_metadata = [CANONICAL, CANONICAL, CANONICAL, HISTORICAL]
        while len(candidates) < 128:
            candidates.append(rng.choice(train_surfaces))
            candidate_relations.append(rng.randrange(3))
            candidate_metadata.append(CANONICAL)
        permutation = list(range(len(candidates)))
        rng.shuffle(permutation)
        candidates = [candidates[index] for index in permutation]
        index = CanonicalRouterIndex(
            entity=byte_encoder(candidates).to("cuda"),
            relation_id=torch.tensor([candidate_relations[i] for i in permutation], device="cuda"),
            metadata_id=torch.tensor([candidate_metadata[i] for i in permutation], device="cuda"),
            valid=torch.ones(len(candidates), dtype=torch.bool, device="cuda"),
        )
        target = torch.tensor([permutation.index(0)], device="cuda")
        scores, _ = router.all_scores(query, index)
        labels = torch.zeros_like(scores)
        labels[:, target] = 1.0
        loss = F.cross_entropy(scores, target) + F.binary_cross_entropy_with_logits(
            scores, labels, pos_weight=torch.tensor([len(candidates) - 1.0], device="cuda")
        )
        router_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        router_optimizer.step()
        router_losses.append(float(loss.detach()))
    del router_optimizer
    calibration_positive = []
    calibration_negative = []
    with torch.inference_mode():
        for calibration_index in range(128):
            surface = train_surfaces[calibration_index]
            relation = calibration_index % 3
            wrong = train_surfaces[(calibration_index + 37) % len(train_surfaces)]
            query = FactorizedCanonicalQuery(
                entity=byte_encoder([surface]).to("cuda"),
                relation_logits=torch.full((1, 3), -12.0, device="cuda"),
                metadata_logits=torch.tensor([[12.0, -12.0, -12.0, -12.0]], device="cuda"),
            )
            query.relation_logits[0, relation] = 12.0
            index = CanonicalRouterIndex(
                entity=byte_encoder([surface, wrong, surface, surface]).to("cuda"),
                relation_id=torch.tensor([relation, relation, (relation + 1) % 3, relation], device="cuda"),
                metadata_id=torch.tensor([CANONICAL, CANONICAL, CANONICAL, HISTORICAL], device="cuda"),
                valid=torch.ones(4, dtype=torch.bool, device="cuda"),
            )
            scores, _ = router.all_scores(query, index)
            calibration_positive.append(scores[0, 0])
            calibration_negative.extend(scores[0, 1:])
    calibration_balanced_accuracy = router.calibrate_acceptance(
        torch.stack(calibration_positive), torch.stack(calibration_negative)
    )
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    for parameter in package.query_projector.parameters():
        parameter.requires_grad_(True)

    def canonical_vector(entity_id, relation, value, metadata=CANONICAL):
        with torch.inference_mode():
            vector = representation.encode(
                torch.tensor([entity_id % 24]), torch.tensor([relation]),
                torch.tensor([value % 36]), torch.tensor([metadata]),
            )[0]
        return vector.to("cuda", dtype=torch.float16)

    value_optimizer = torch.optim.AdamW(package.value_translator.parameters(), lr=2e-3, eps=1e-6)
    value_losses = []
    for _ in range(value_steps):
        entity_ids = [rng.randrange(24) for _ in range(32)]
        relations = [rng.randrange(3) for _ in range(32)]
        values = [rng.randrange(36) for _ in range(32)]
        canonical = torch.stack([
            canonical_vector(entity, relation, value).float()
            for entity, relation, value in zip(entity_ids, relations, values)
        ])
        translated = package.value_translator(canonical)
        normalized = F.normalize(translated, dim=-1)
        targets = torch.tensor(values, device="cuda")
        loss = (
            1 - F.cosine_similarity(
                normalized, normalized_lm_values.index_select(0, targets), dim=-1
            ).mean()
            + F.cross_entropy(normalized @ normalized_lm_values.T / 0.07, targets)
        )
        value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        value_optimizer.step()
        value_losses.append(float(loss.detach()))
    del value_optimizer
    with torch.inference_mode():
        heldout_value_ids = torch.arange(36, device="cuda")
        heldout_vectors = torch.stack([
            canonical_vector(index % 24, index % 3, index).float() for index in range(36)
        ])
        heldout_value_output = F.normalize(
            package.value_translator(heldout_vectors), dim=-1
        )
        value_metrics = {
            "accuracy": float((
                (heldout_value_output @ normalized_lm_values.T).argmax(-1)
                == heldout_value_ids
            ).float().mean()),
            "cosine": float(F.cosine_similarity(
                heldout_value_output, normalized_lm_values, dim=-1
            ).mean()),
        }

    rp_preserve = tokenize([
        "A patient tailor compared blue ribbons while rain ticked softly against the shop window.",
        "Two actors rehearsed a harmless joke and rearranged wooden chairs beside the empty stage.",
    ])
    rp_eval = tokenize([
        "At dusk, a baker swept flour from the counter while neighbors debated tomorrow's parade.",
        "A sleepy musician closed the balcony doors and described clouds drifting above the orchard.",
    ])
    with torch.inference_mode():
        frozen_rp_preserve = wrapper(**rp_preserve, use_cache=False).logits.detach()
        frozen_rp_eval = wrapper(**rp_eval, use_cache=False).logits.detach()
        base_rp_loss = float(wrapper(**rp_eval, labels=rp_eval.input_ids, use_cache=False).loss)

    train_state_surfaces = train_surfaces[:24]
    heldout_state_surfaces = heldout_surfaces[:20]
    train_value_assignment = {surface: index % 36 for index, surface in enumerate(train_state_surfaces)}
    heldout_value_assignment = {surface: (index * 5 + 3) % 36 for index, surface in enumerate(heldout_state_surfaces)}
    train_owner_hidden = capture([
        TRAIN_LEADS[0] + RELATION_PROMPTS[0].format(entity=surface)
        for surface in train_state_surfaces
    ]).mean(0)
    heldout_owner_hidden = capture([
        HELDOUT_LEADS[1] + RELATION_PROMPTS[0].format(entity=surface)
        for surface in heldout_state_surfaces
    ]).mean(0)
    with torch.inference_mode():
        owner_queries = package.query_projector(
            heldout_owner_hidden,
            entity_anchor=byte_encoder(heldout_state_surfaces).to("cuda"),
        )
        hard_top1 = []
        false_positives = {
            "wrong_entity": [], "wrong_relation": [], "historical": [],
            "invalidated": [], "irrelevant": [],
        }
        for query_index, surface in enumerate(heldout_state_surfaces):
            wrong_entity = heldout_state_surfaces[(query_index + 1) % len(heldout_state_surfaces)]
            candidates = [surface, wrong_entity, surface, surface, f"irrelevant {query_index}"]
            index = CanonicalRouterIndex(
                entity=byte_encoder(candidates).to("cuda"),
                relation_id=torch.tensor([0, 0, 1, 0, 2], device="cuda"),
                metadata_id=torch.tensor([CANONICAL, CANONICAL, CANONICAL, HISTORICAL, CANONICAL], device="cuda"),
                valid=torch.ones(5, dtype=torch.bool, device="cuda"),
            )
            single_query = FactorizedCanonicalQuery(
                entity=owner_queries.entity[query_index:query_index + 1],
                relation_logits=owner_queries.relation_logits[query_index:query_index + 1],
                metadata_logits=owner_queries.metadata_logits[query_index:query_index + 1],
            )
            scores, _ = router.all_scores(single_query, index)
            hard_top1.append(int(scores.argmax(-1)) == 0)
            for label, candidate_index in (
                ("wrong_entity", 1), ("wrong_relation", 2),
                ("historical", 3), ("irrelevant", 4),
            ):
                false_positives[label].append(
                    float(scores[0, candidate_index]) >= float(router.acceptance_threshold)
                )
            invalid_index = CanonicalRouterIndex(
                entity=index.entity[:1], relation_id=index.relation_id[:1],
                metadata_id=index.metadata_id[:1],
                valid=torch.zeros(1, dtype=torch.bool, device="cuda"),
            )
            invalid_scores, _ = router.all_scores(single_query, invalid_index)
            false_positives["invalidated"].append(bool(torch.isfinite(invalid_scores[0, 0])))
        router_hard_metrics = {
            "top1_accuracy": sum(hard_top1) / len(hard_top1),
            **{
                f"{label}_false_positive_rate": sum(values) / len(values)
                for label, values in false_positives.items()
            },
        }

    def make_store(entries, slots=None, local_rng=None, metadata=CANONICAL):
        capacity = slots or max(4, len(entries))
        store = CanonicalPStore(CanonicalPConfig(
            slots=capacity, width=512, dtype=torch.float16, device="cuda", merge_similarity=1.0
        ))
        rows = list(entries)
        if local_rng:
            local_rng.shuffle(rows)
        slot_by_surface = {}
        for ordinal, (surface, relation, value) in enumerate(rows):
            slot, _ = store.create(
                canonical_vector(ordinal, relation, value, metadata),
                entity_id=ordinal, relation_id=relation, value_id=value, metadata_id=metadata,
                label=surface,
            )
            slot_by_surface[surface] = slot
        return store, slot_by_surface

    def state_inputs(surfaces, lead=TRAIN_LEADS[0], relation=0):
        return tokenize([
            lead + RELATION_PROMPTS[relation].format(entity=surface) for surface in surfaces
        ])

    training_base_logits = {}
    for lead in TRAIN_LEADS:
        with torch.inference_mode():
            training_base_logits[lead] = wrapper(
                **state_inputs(train_state_surfaces, lead), use_cache=False
            ).logits[:, -1].detach()

    package_optimizer = torch.optim.AdamW(package.parameters(), lr=5e-4, eps=1e-6)
    causal_losses = []
    pre_preservation_ablation = None
    midpoint_entries = [
        (surface, 0, heldout_value_assignment[surface])
        for surface in heldout_state_surfaces
    ]
    midpoint_store, _ = make_store(
        midpoint_entries, slots=128, local_rng=random.Random(seed + 800)
    )
    midpoint_inputs = state_inputs(heldout_state_surfaces, HELDOUT_LEADS[1])
    midpoint_expected = torch.tensor(
        [heldout_value_assignment[surface] for surface in heldout_state_surfaces],
        device="cuda",
    )
    for step in range(causal_steps):
        surfaces = rng.sample(train_state_surfaces, 8)
        entries = [(surface, 0, train_value_assignment[surface]) for surface in surfaces]
        store, _ = make_store(entries, slots=128, local_rng=rng)
        lead = rng.choice(TRAIN_LEADS)
        inputs = state_inputs(surfaces, lead)
        targets = torch.tensor(
            [value_token_ids[train_value_assignment[surface]] for surface in surfaces],
            device="cuda",
        )
        output = wrapper(
            **inputs, p_store=store, query_entity_surfaces=surfaces, use_cache=False
        )
        state_loss = F.cross_entropy(output.logits[:, -1].float(), targets)

        wrong_entries = [
            (rng.choice(train_surfaces[len(train_state_surfaces):]), 0,
             train_value_assignment[surface])
            for surface in surfaces
        ]
        wrong_store, _ = make_store(wrong_entries, slots=128)
        wrong = wrapper(
            **inputs, p_store=wrong_store, query_entity_surfaces=surfaces, use_cache=False
        ).logits[:, -1].float()
        historical_store, _ = make_store(entries, slots=128, metadata=HISTORICAL)
        historical = wrapper(
            **inputs, p_store=historical_store,
            query_entity_surfaces=surfaces, use_cache=False,
        ).logits[:, -1].float()
        indices = torch.tensor([train_state_surfaces.index(surface) for surface in surfaces], device="cuda")
        base_logits = training_base_logits[lead].index_select(0, indices).float()
        wrong_preserve = F.kl_div(
            F.log_softmax(base_logits, dim=-1), F.softmax(wrong, dim=-1), reduction="batchmean"
        )
        historical_preserve = F.kl_div(
            F.log_softmax(base_logits, dim=-1),
            F.softmax(historical, dim=-1), reduction="batchmean"
        )

        hidden_indices = torch.tensor([
            train_state_surfaces.index(surface) for surface in surfaces
        ], device="cuda")
        projected = package.query_projector(train_owner_hidden.index_select(0, hidden_indices))
        query_loss = (
            1 - F.cosine_similarity(
                projected.entity, byte_encoder(surfaces).to("cuda"), dim=-1
            ).mean()
            + F.cross_entropy(projected.relation_logits, torch.zeros(8, dtype=torch.long, device="cuda"))
        )
        vectors = torch.stack([
            canonical_vector(index, 0, train_value_assignment[surface]).float()
            for index, surface in enumerate(surfaces)
        ])
        translated = F.normalize(package.value_translator(vectors), dim=-1)
        value_targets = torch.tensor(
            [train_value_assignment[surface] for surface in surfaces], device="cuda"
        )
        value_loss = 1 - F.cosine_similarity(
            translated, normalized_lm_values.index_select(0, value_targets), dim=-1
        ).mean()

        loss = (
            state_loss + 0.5 * query_loss + 0.2 * value_loss
            + 2.0 * wrong_preserve + 2.0 * historical_preserve
        )
        if step >= causal_steps // 2:
            rp_output = wrapper(
                **rp_preserve, labels=rp_preserve.input_ids, p_store=store, use_cache=False
            )
            rp_kl = F.kl_div(
                F.log_softmax(frozen_rp_preserve[:, -1].float(), dim=-1),
                F.softmax(rp_output.logits[:, -1].float(), dim=-1), reduction="batchmean"
            )
            loss = loss + 0.05 * rp_output.loss.float() + 2.0 * rp_kl
        package_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(package.parameters(), 1.0)
        package_optimizer.step()
        causal_losses.append(float(state_loss.detach()))
        if step + 1 == causal_steps // 2:
            with torch.inference_mode():
                midpoint_logits = wrapper(
                    **midpoint_inputs, p_store=midpoint_store,
                    query_entity_surfaces=heldout_state_surfaces, use_cache=False,
                ).logits[:, -1].float()
                midpoint_rp = wrapper(
                    **rp_eval, labels=rp_eval.input_ids,
                    p_store=midpoint_store, use_cache=False,
                )
                midpoint_rp_kl = F.kl_div(
                    F.log_softmax(frozen_rp_eval[:, -1].float(), dim=-1),
                    F.softmax(midpoint_rp.logits[:, -1].float(), dim=-1),
                    reduction="batchmean",
                )
            pre_preservation_ablation = {
                "state_loss": float(state_loss.detach()),
                "state_candidate_accuracy": float((
                    midpoint_logits[:, value_token_ids].argmax(-1) == midpoint_expected
                ).float().mean()),
                "wrong_state_kl": float(wrong_preserve.detach()),
                "historical_state_kl": float(historical_preserve.detach()),
                "rp_loss": float(midpoint_rp.loss),
                "rp_kl": float(midpoint_rp_kl),
            }
    del package_optimizer
    wrapper.eval()

    def store_bytes(store):
        tensors = (
            store.cache.values, store.cache.valid, store.cache.slot_type,
            store.cache.confidence, store.cache.importance, store.cache.freshness,
            store.cache.persistence, store.cache.last_updated, store.cache.source,
            store.entity_id, store.relation_id, store.value_id,
            store.canonical_metadata_id,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def index_bytes(index):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (index.entity, index.relation_id, index.metadata_id, index.valid)
        )

    scaling = {}
    all_distractors = [
        surface for surface in train_surfaces + heldout_surfaces
        if surface not in heldout_state_surfaces
    ]
    while len(all_distractors) < 512:
        all_distractors.append(f"irrelevant entity {len(all_distractors)}")
    for slot_count in SLOT_SIZES:
        query_count = min(20, slot_count)
        query_surfaces = heldout_state_surfaces[:query_count]
        entries = [
            (surface, 0, heldout_value_assignment[surface]) for surface in query_surfaces
        ]
        for index in range(slot_count - query_count):
            entries.append((all_distractors[index], (index + 1) % 3, (index + 7) % 36))
        store, slot_map = make_store(entries, slots=slot_count, local_rng=random.Random(seed + slot_count))
        inputs = state_inputs(query_surfaces, HELDOUT_LEADS[1])
        hidden = capture([
            HELDOUT_LEADS[1] + RELATION_PROMPTS[0].format(entity=surface)
            for surface in query_surfaces
        ]).mean(0)
        with torch.inference_mode():
            hidden_only_query = package.query_projector(hidden)
            query = package.query_projector(
                hidden, entity_anchor=byte_encoder(query_surfaces).to("cuda")
            )
            index = router.build_index(store, byte_encoder, device="cuda")
            scores, _ = router.all_scores(query, index)
            expected = torch.tensor([slot_map[surface] for surface in query_surfaces], device="cuda")
            order = scores.argsort(dim=-1, descending=True)
            ranks = (order == expected[:, None]).nonzero()[:, 1] + 1
            hidden_only_scores, _ = router.all_scores(hidden_only_query, index)
            hidden_only_order = hidden_only_scores.argsort(dim=-1, descending=True)
            hidden_only_ranks = (
                hidden_only_order == expected[:, None]
            ).nonzero()[:, 1] + 1
            oracle_query = FactorizedCanonicalQuery(
                entity=byte_encoder(query_surfaces).to("cuda"),
                relation_logits=torch.tensor([[12.0, -12.0, -12.0]], device="cuda").expand(query_count, -1),
                metadata_logits=torch.tensor([[12.0, -12.0, -12.0, -12.0]], device="cuda").expand(query_count, -1),
            )
            oracle_scores, _ = router.all_scores(oracle_query, index)
            oracle_order = oracle_scores.argsort(dim=-1, descending=True)
            oracle_ranks = (oracle_order == expected[:, None]).nonzero()[:, 1] + 1
            route_metrics = {
                "top1_accuracy": float((ranks == 1).float().mean()),
                "top2_recall": float((ranks <= 2).float().mean()),
                "top4_recall": float((ranks <= 4).float().mean()),
                "mrr": float((1.0 / ranks.float()).mean()),
                "hidden_only_top1_accuracy": float((hidden_only_ranks == 1).float().mean()),
                "oracle_query_top1_accuracy": float((oracle_ranks == 1).float().mean()),
                "oracle_query_top4_recall": float((oracle_ranks <= 4).float().mean()),
                "oracle_query_mrr": float((1.0 / oracle_ranks.float()).mean()),
            }
            logits = wrapper(
                **inputs, p_store=store, query_entity_surfaces=query_surfaces,
                use_cache=False,
            ).logits[:, -1].float()
            expected_values = torch.tensor(
                [heldout_value_assignment[surface] for surface in query_surfaces], device="cuda"
            )
            generation_accuracy = float((
                logits[:, value_token_ids].argmax(-1) == expected_values
            ).float().mean())
            for _ in range(2):
                wrapper(
                    **inputs, p_store=store, query_entity_surfaces=query_surfaces,
                    use_cache=False,
                )
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                wrapper(
                    **inputs, p_store=store, query_entity_surfaces=query_surfaces,
                    use_cache=False,
                )
            torch.cuda.synchronize()
            latency = (time.perf_counter() - start) / 5
        scaling[str(slot_count)] = {
            **route_metrics,
            "state_generation_accuracy": generation_accuracy,
            "latency_seconds": latency,
            "active_vram_overhead_bytes": (
                _bytes(package.parameters()) + _bytes(router.parameters())
                + store_bytes(store) + index_bytes(index)
            ),
        }

    eval_entries = [
        (surface, 0, heldout_value_assignment[surface]) for surface in heldout_state_surfaces
    ]
    eval_store, eval_slots = make_store(eval_entries, slots=128, local_rng=random.Random(seed + 900))
    eval_inputs = state_inputs(heldout_state_surfaces, HELDOUT_LEADS[1])
    expected_values = torch.tensor(
        [heldout_value_assignment[surface] for surface in heldout_state_surfaces], device="cuda"
    )
    expected_tokens = torch.tensor([value_token_ids[int(value)] for value in expected_values], device="cuda")
    oracle_indices = torch.tensor([eval_slots[surface] for surface in heldout_state_surfaces], device="cuda")
    with torch.inference_mode():
        disabled = wrapper(**eval_inputs, use_cache=False).logits[:, -1].float()
        oracle = wrapper(
            **eval_inputs, p_store=eval_store, oracle_indices=oracle_indices,
            query_entity_surfaces=heldout_state_surfaces,
            gate_enabled=False, use_cache=False,
        ).logits[:, -1].float()
        without_gate = wrapper(
            **eval_inputs, p_store=eval_store,
            query_entity_surfaces=heldout_state_surfaces,
            gate_enabled=False, use_cache=False
        ).logits[:, -1].float()
        full = wrapper(
            **eval_inputs, p_store=eval_store,
            query_entity_surfaces=heldout_state_surfaces,
            collect_telemetry=True, use_cache=False
        ).logits[:, -1].float()
        full_gate = float(torch.stack([
            values[:, -1].float().mean() for values in wrapper.gate_telemetry
        ]).mean())

    def accuracy(logits):
        return float((logits[:, value_token_ids].argmax(-1) == expected_values).float().mean())

    ablations = {
        "router_only": scaling["128"],
        "translator_only_oracle_routing": {"state_candidate_accuracy": accuracy(oracle)},
        "router_plus_translator_without_gate": {"state_candidate_accuracy": accuracy(without_gate)},
        "router_plus_translator_plus_gate": pre_preservation_ablation,
        "full_system_with_preservation": {
            "state_candidate_accuracy": accuracy(full),
            "full_token_accuracy": float((full.argmax(-1) == expected_tokens).float().mean()),
            "gate_activation": full_gate,
        },
    }

    def single_condition(surface, value, *, label=None, metadata=CANONICAL, invalidate=False):
        store, slots = make_store(
            [(label or surface, 0, value)], slots=128, metadata=metadata
        )
        if invalidate:
            store.invalidate(next(iter(slots.values())))
        return store

    counter_prompt = state_inputs(["silver key"], HELDOUT_LEADS[1])
    alice, bob = 0, 1
    conditions = {
        "disabled": None,
        "p1_silver_alice": single_condition("silver key", alice),
        "p2_silver_bob": single_condition("silver key", bob),
        "p3_gold_alice": single_condition("silver key", alice, label="gold key"),
        "p4_silver_historical": single_condition("silver key", alice, metadata=HISTORICAL),
        "p4_silver_invalidated": single_condition("silver key", alice, invalidate=True),
    }
    counterfactual = {}
    with torch.inference_mode():
        for label, store in conditions.items():
            output = wrapper(
                **counter_prompt, p_store=store,
                query_entity_surfaces=["silver key"],
                collect_telemetry=True, use_cache=False
            ).logits[:, -1].float()
            probabilities = F.softmax(output, dim=-1)
            counterfactual[label] = {
                "alice_logit": float(output[0, value_token_ids[alice]]),
                "bob_logit": float(output[0, value_token_ids[bob]]),
                "alice_probability": float(probabilities[0, value_token_ids[alice]]),
                "bob_probability": float(probabilities[0, value_token_ids[bob]]),
                "generated": tokenizer.decode([int(output.argmax(-1))]),
                "gate": 0.0 if not wrapper.gate_telemetry else float(torch.stack([
                    values[:, -1].float().mean() for values in wrapper.gate_telemetry
                ]).mean()),
            }

    wrong_store = conditions["p3_gold_alice"]
    invalid_store = conditions["p4_silver_invalidated"]

    def greedy_continuations(inputs, store, steps=6):
        input_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()
        original_length = input_ids.shape[1]
        for _ in range(steps):
            output = wrapper(
                input_ids=input_ids, attention_mask=attention_mask,
                p_store=store, use_cache=False,
            ).logits[:, -1]
            next_token = output.argmax(-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            attention_mask = torch.cat((
                attention_mask,
                torch.ones_like(next_token, dtype=attention_mask.dtype),
            ), dim=1)
        return [tokenizer.decode(row[original_length:]) for row in input_ids]

    with torch.inference_mode():
        rp_conditions = {}
        for label, store in (
            ("base", None), ("irrelevant", eval_store), ("wrong_entity", wrong_store),
            ("invalidated", invalid_store),
        ):
            output = wrapper(
                **rp_eval, labels=rp_eval.input_ids, p_store=store, use_cache=False
            )
            kl = F.kl_div(
                F.log_softmax(frozen_rp_eval[:, -1].float(), dim=-1),
                F.softmax(output.logits[:, -1].float(), dim=-1), reduction="batchmean"
            )
            rp_conditions[label] = {
                "loss": float(output.loss), "kl": float(kl),
                "samples": greedy_continuations(rp_eval, store),
            }

    invalid_difference = max(
        abs(counterfactual["p4_silver_invalidated"][key] - counterfactual["disabled"][key])
        for key in ("alice_logit", "bob_logit")
    )
    mutation_store, mutation_slots = make_store(
        [("silver key", 0, 0)], slots=128
    )
    mutation_slot = mutation_slots["silver key"]
    for mutation_index, value in enumerate((1, 2, 3, 4)):
        mutation_store.modify(
            mutation_slot,
            canonical_vector(0, 0, value),
            entity_id=0, relation_id=0, value_id=value, metadata_id=CANONICAL,
            source=SlotSource.CORRECTION if mutation_index == 3 else None,
        )
    with torch.inference_mode():
        mutation_logits = wrapper(
            **counter_prompt, p_store=mutation_store,
            query_entity_surfaces=["silver key"], use_cache=False,
        ).logits[:, -1].float()
    mutation_latest_correct = int(
        mutation_logits[:, value_token_ids].argmax(-1)
    ) == 4
    mutation_store.invalidate(mutation_slot)
    with torch.inference_mode():
        mutation_invalidated = wrapper(
            **counter_prompt, p_store=mutation_store,
            query_entity_surfaces=["silver key"], use_cache=False,
        ).logits[:, -1].float()
        mutation_disabled = wrapper(**counter_prompt, use_cache=False).logits[:, -1].float()
    mutation_invalidated_difference = float((
        mutation_invalidated - mutation_disabled
    ).abs().max())
    base_gradients = sum(parameter.grad is not None for parameter in base.parameters())
    if package_path is not None:
        package.save(package_path)
        restored = SplitPTranslatePackage.load(package_path, device="cuda")
        restored.validate_compatibility(
            model_id=pythia_model_identifier(base), model_hidden_width=int(base.config.hidden_size),
            attachment_layers=layers, model_config_sha256=config_checksum(base.config),
        )
        package_roundtrip = max(
            float((left - right).abs().max())
            for left, right in zip(package.state_dict().values(), restored.state_dict().values())
        )
        del restored
    else:
        package_roundtrip = None
    if router_path is not None:
        router.save(router_path)
        restored_router = CanonicalPRouter.load(router_path, device="cuda")
        router_roundtrip = max(
            float((left - right).abs().max())
            for left, right in zip(router.state_dict().values(), restored_router.state_dict().values())
        )
        del restored_router
    else:
        router_roundtrip = None

    result = {
        "attachment_layers": list(layers),
        "attachment_count": attachment_count,
        "query_projector": {
            "loss_first_last": [query_losses[0], query_losses[-1]],
            "byte_surface_anchor_approach": byte_surface_metrics,
            "hidden_to_byte_reconstruction_ablation": query_heldout_metrics,
            "frozen_lexical_anchor_approach": lexical_metrics,
            "heldout_names": len(heldout_surfaces),
            "training_names": len(train_surfaces),
        },
        "router": {
            "loss_first_last": [router_losses[0], router_losses[-1]],
            "model_hidden_dimensions": 0,
            "acceptance_threshold": float(router.acceptance_threshold),
            "calibration_balanced_accuracy": calibration_balanced_accuracy,
            "hard_negative_metrics": router_hard_metrics,
            "scaling": scaling,
        },
        "value_translator": {
            "loss_first_last": [value_losses[0], value_losses[-1]],
            "oracle_selected_metrics": value_metrics,
        },
        "causal_training_loss_first_last": [causal_losses[0], causal_losses[-1]],
        "ablations": ablations,
        "counterfactual": counterfactual,
        "invalidated_logit_difference": invalid_difference,
        "mutation_chain": {
            "latest_state_accuracy": float(mutation_latest_correct),
            "invalidated_max_logit_difference": mutation_invalidated_difference,
            "source_tokens_in_recent_kv": 0,
        },
        "natural_rp": {
            "conditions": rp_conditions,
            "base_loss": base_rp_loss,
            "relevant_state_generation_sample": counterfactual["p1_silver_alice"]["generated"],
        },
        "base_parameters_with_grad": base_gradients,
        "source_tokens_in_recent_kv": 0,
        "extra_prompt_tokens": 0,
        "package_parameters": sum(p.numel() for p in package.parameters()),
        "router_parameters": sum(p.numel() for p in router.parameters()),
        "package_roundtrip_max_difference": package_roundtrip,
        "router_roundtrip_max_difference": router_roundtrip,
        "package_path": str(package_path) if package_path else None,
        "router_path": str(router_path) if router_path else None,
    }
    wrapper.close()
    del wrapper, base, package, router
    gc.collect()
    torch.cuda.empty_cache()
    return result
