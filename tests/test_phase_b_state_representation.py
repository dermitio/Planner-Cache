import random

import torch

from pcm.planner.representation import (
    CompositionConfig,
    composition_splits,
    is_held_out,
    make_batch,
    train_and_probe_representation,
)


def test_composition_split_holds_out_pairs_not_entities_or_values():
    config = CompositionConfig(entities=8, relations=2, values=12)
    train, held_out = composition_splits(config)
    assert all(not is_held_out(*row) for row in train)
    assert all(is_held_out(*row) for row in held_out)
    assert {row[0] for row in train} == {row[0] for row in held_out}
    assert {row[2] for row in train} == {row[2] for row in held_out}


def test_candidate_slot_order_is_randomized_and_contains_hard_negatives():
    config = CompositionConfig(entities=8, relations=2, values=12)
    train, _ = composition_splits(config)
    positions = set()
    kinds_seen = set()
    for seed in range(12):
        _, _, target, kinds = make_batch(
            train[:1], config, 1, random.Random(seed), held_out=False
        )
        positions.add(int(target[0]))
        kinds_seen.update(kinds[0])
    assert len(positions) > 1
    assert kinds_seen == {"correct", "wrong_value", "wrong_entity", "historical"}


def test_factorized_contrastive_representation_recovers_heldout_compositions():
    _, _, result = train_and_probe_representation(
        steps=350, batch_size=48, slot_width=128, evaluation_limit=96
    )
    assert result["training_loss_last"] < result["training_loss_first"] * 0.1
    assert result["p_only_state_recovery"] >= 0.9
    assert result["permutation_stability"] >= 0.9
    assert min(result["hard_negative_accuracy"].values()) >= 0.9
    assert min(result["canonical_decode_accuracy"].values()) >= 0.9
