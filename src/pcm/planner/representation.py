"""Factorized, contrastively trained canonical state representation probes."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CANONICAL = 0
CONTRADICTED = 1
HISTORICAL = 2
INFERRED = 3


class FactorizedStateRepresentation(nn.Module):
    """Separate entity/relation/value/metadata fields projected to a P slot."""

    def __init__(
        self,
        entities: int,
        relations: int,
        values: int,
        metadata: int = 4,
        *,
        field_width: int = 128,
        slot_width: int = 512,
    ) -> None:
        super().__init__()
        self.entity = nn.Embedding(entities, field_width)
        self.relation = nn.Embedding(relations, field_width)
        self.value = nn.Embedding(values, field_width)
        self.metadata = nn.Embedding(metadata, field_width)
        factor_width = field_width * 4
        self.slot_projection = nn.Linear(factor_width, slot_width, bias=False)
        self.tuple_projection = nn.Linear(factor_width, slot_width, bias=False)
        self.query_projection = nn.Linear(field_width * 3, slot_width, bias=False)
        self.entity_decoder = nn.Linear(slot_width, entities)
        self.relation_decoder = nn.Linear(slot_width, relations)
        self.value_decoder = nn.Linear(slot_width, values)
        self.metadata_decoder = nn.Linear(slot_width, metadata)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def factors(self, entity: Tensor, relation: Tensor, value: Tensor, metadata: Tensor) -> Tensor:
        return torch.cat((
            self.entity(entity),
            self.relation(relation),
            self.value(value),
            self.metadata(metadata),
        ), dim=-1)

    def encode(self, entity: Tensor, relation: Tensor, value: Tensor, metadata: Tensor) -> Tensor:
        return F.normalize(self.slot_projection(self.factors(entity, relation, value, metadata)), dim=-1)

    def tuple_anchor(self, entity: Tensor, relation: Tensor, value: Tensor, metadata: Tensor) -> Tensor:
        return F.normalize(self.tuple_projection(self.factors(entity, relation, value, metadata)), dim=-1)

    def query(self, entity: Tensor, relation: Tensor, metadata: Tensor) -> Tensor:
        fields = torch.cat((self.entity(entity), self.relation(relation), self.metadata(metadata)), dim=-1)
        return F.normalize(self.query_projection(fields), dim=-1)

    def scores(self, query: Tensor, slots: Tensor) -> Tensor:
        temperature = self.temperature.clamp(0.02, 1.0)
        return torch.einsum("bd,bkd->bk", query, slots) / temperature

    def decode(self, slots: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.entity_decoder(slots),
            self.relation_decoder(slots),
            self.value_decoder(slots),
            self.metadata_decoder(slots),
        )


@dataclass(frozen=True)
class CompositionConfig:
    entities: int = 24
    relations: int = 3
    values: int = 36
    candidates: int = 4


def is_held_out(entity: int, relation: int, value: int) -> bool:
    return (entity * 31 + relation * 17 + value * 13) % 5 == 0


def composition_splits(config: CompositionConfig):
    train, held_out = [], []
    for entity in range(config.entities):
        for relation in range(config.relations):
            for value in range(config.values):
                target = held_out if is_held_out(entity, relation, value) else train
                target.append((entity, relation, value))
    return train, held_out


def _same_split_alternative(entity, relation, value, size, want_held_out, field):
    for offset in range(1, size):
        if field == "value":
            candidate = (entity, relation, (value + offset) % size)
        else:
            candidate = ((entity + offset) % size, relation, value)
        if is_held_out(*candidate) == want_held_out:
            return candidate
    raise RuntimeError("unable to construct composition-preserving negative")


def candidate_tuples(positive, config: CompositionConfig, *, held_out: bool):
    entity, relation, value = positive
    wrong_value = _same_split_alternative(
        entity, relation, value, config.values, held_out, "value"
    )
    wrong_entity = _same_split_alternative(
        entity, relation, value, config.entities, held_out, "entity"
    )
    return (
        (entity, relation, value, CANONICAL, "correct"),
        (*wrong_value, CONTRADICTED, "wrong_value"),
        (*wrong_entity, CANONICAL, "wrong_entity"),
        (entity, relation, value, HISTORICAL, "historical"),
    )


def make_batch(combinations, config, batch_size, rng, *, held_out, permute=True):
    selected = [combinations[rng.randrange(len(combinations))] for _ in range(batch_size)]
    candidates, targets, kinds = [], [], []
    for positive in selected:
        rows = list(candidate_tuples(positive, config, held_out=held_out))
        if permute:
            rng.shuffle(rows)
        candidates.append([row[:4] for row in rows])
        kinds.append([row[4] for row in rows])
        targets.append(next(index for index, row in enumerate(rows) if row[4] == "correct"))
    positive = torch.tensor(selected, dtype=torch.long)
    return positive, torch.tensor(candidates, dtype=torch.long), torch.tensor(targets), kinds


def representation_loss(model, positive, candidates, targets):
    entity, relation, value = positive.T
    flat = candidates.view(-1, 4)
    slots = model.encode(*flat.T).view(candidates.shape[0], candidates.shape[1], -1)
    query = model.query(entity, relation, torch.full_like(entity, CANONICAL))
    retrieval = F.cross_entropy(model.scores(query, slots), targets)
    anchor = model.tuple_anchor(entity, relation, value, torch.full_like(entity, CANONICAL))
    contrastive = F.cross_entropy(model.scores(anchor, slots), targets)
    decoded = model.decode(slots)
    canonical = sum(
        F.cross_entropy(logits.flatten(0, 1), flat[:, field])
        for field, logits in enumerate(decoded)
    )
    return retrieval + contrastive + 0.5 * canonical


def evaluate_representation(model, combinations, config, *, permutations=8, seed=101):
    model.eval()
    totals = {"correct": 0, "wrong_value": 0, "wrong_entity": 0, "historical": 0}
    count = 0
    decoded = torch.zeros(4)
    stable = 0
    rng = random.Random(seed)
    with torch.inference_mode():
        for positive in combinations:
            chosen_values = []
            for _ in range(permutations):
                pos, candidates, target, kinds = make_batch(
                    [positive], config, 1, rng, held_out=True, permute=True
                )
                flat = candidates.view(-1, 4)
                slots = model.encode(*flat.T).view(1, config.candidates, -1)
                query = model.query(pos[:, 0], pos[:, 1], torch.zeros(1, dtype=torch.long))
                scores = model.scores(query, slots)[0]
                correct_index = int(target[0])
                prediction = int(scores.argmax())
                selected_slot = slots[0, prediction]
                selected_value = int(model.value_decoder(selected_slot).argmax())
                chosen_values.append(selected_value)
                if prediction == correct_index and selected_value == positive[2]:
                    totals["correct"] += 1
                for index, kind in enumerate(kinds[0]):
                    if kind != "correct" and scores[correct_index] > scores[index]:
                        totals[kind] += 1
                decoded_logits = model.decode(slots[0, correct_index])
                truth = candidates[0, correct_index]
                decoded += torch.tensor([
                    int(logits.argmax() == truth[field])
                    for field, logits in enumerate(decoded_logits)
                ])
                count += 1
            stable += int(len(set(chosen_values)) == 1 and chosen_values[0] == positive[2])
    return {
        "p_only_state_recovery": totals["correct"] / count,
        "hard_negative_accuracy": {
            kind: totals[kind] / count for kind in ("wrong_value", "wrong_entity", "historical")
        },
        "canonical_decode_accuracy": {
            field: float(decoded[index] / count)
            for index, field in enumerate(("entity", "relation", "value", "metadata"))
        },
        "permutation_stability": stable / len(combinations),
        "held_out_combinations": len(combinations),
        "permutations_per_combination": permutations,
    }


def train_and_probe_representation(
    *, steps=600, batch_size=64, slot_width=512, seed=97, evaluation_limit=256
):
    torch.manual_seed(seed)
    config = CompositionConfig()
    train, held_out = composition_splits(config)
    model = FactorizedStateRepresentation(
        config.entities, config.relations, config.values, slot_width=slot_width
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed)
    losses = []
    model.train()
    for _ in range(steps):
        positive, candidates, targets, _ = make_batch(
            train, config, batch_size, rng, held_out=False
        )
        loss = representation_loss(model, positive, candidates, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    probe = evaluate_representation(
        model, held_out[:evaluation_limit], config, permutations=8, seed=seed + 1
    )
    probe.update({
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "train_combinations": len(train),
        "total_held_out_combinations": len(held_out),
        "slot_width": slot_width,
    })
    return model, config, probe
