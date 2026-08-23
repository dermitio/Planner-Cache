"""Shared interactive session state, event recording, and conservative extraction."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Iterable

import torch

from pcm.planner.cache import (
    Freshness,
    Persistence,
    SlotSource,
    SlotType,
    StateOperation,
)
from pcm.planner.canonical import (
    CANONICAL_P_PROTOCOL,
    CANONICAL_VALUE_LABELS,
    CanonicalPConfig,
    CanonicalPStore,
)
from pcm.planner.personality import (
    EvidenceAuthority,
    EvidenceRecord,
    FactorizedPersonalityCanonicalizer,
    PersonalityPackage,
    PersonalityQuery,
    PersonalityRouter,
    PersonalityStatus,
    PersonalityType,
)
from pcm.planner.representation import CANONICAL, FactorizedStateRepresentation


EventSink = Callable[[str], None]
RELATION_NAMES = ("owner", "location", "status")
VALUE_LOOKUP = {
    label.casefold(): (index, label)
    for index, label in enumerate(CANONICAL_VALUE_LABELS)
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def safe_session_stamp(stamp: str) -> str:
    return stamp.replace(":", "").replace("+", "p").replace(".", "-")


class SessionRecorder:
    """Append-only JSONL recorder with a human-readable final snapshot."""

    def __init__(
        self,
        session_root: str | Path,
        runtime_name: str,
        metadata: dict[str, object],
        *,
        enabled: bool = True,
        started_at: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.started_at = started_at or utc_now()
        self.started_monotonic = time.monotonic()
        self.runtime_name = runtime_name
        self.turn = 0
        self.event_counts: Counter[str] = Counter()
        self.recent_events: deque[dict[str, object]] = deque(maxlen=50)
        self.metadata = dict(metadata)
        self.metadata.update({
            "session_id": f"{safe_session_stamp(self.started_at)}-{runtime_name}",
            "start_timestamp": self.started_at,
        })
        root = Path(session_root)
        candidate = root / str(self.metadata["session_id"])
        suffix = 1
        while candidate.exists():
            candidate = root / f"{self.metadata['session_id']}-{suffix}"
            suffix += 1
        self.directory = candidate
        self.transcript_path = candidate / "transcript.jsonl"
        self.events_path = candidate / "events.jsonl"
        self.session_path = candidate / "session.json"
        self.final_state_path = candidate / "final-state.json"
        self._transcript_handle = None
        self._events_handle = None
        self._closed = False
        if enabled:
            candidate.mkdir(parents=True, exist_ok=False)
            self._transcript_handle = self.transcript_path.open("a", encoding="utf-8")
            self._events_handle = self.events_path.open("a", encoding="utf-8")
            self.session_path.write_text(
                json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.event("SESSION_START", source="session", metadata=self.metadata)

    def event(self, event: str, *, turn: int | None = None, **fields: object) -> dict[str, object]:
        row = {
            "timestamp": utc_now(),
            "turn": self.turn if turn is None else turn,
            "event": event,
            **fields,
        }
        self.event_counts[event] += 1
        self.recent_events.append(row)
        if self.enabled and self._events_handle is not None:
            self._events_handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            self._events_handle.flush()
        return row

    def transcript(
        self,
        *,
        role: str,
        text: str,
        model: str,
        runtime: str,
        latency_seconds: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        row = {
            "timestamp": utc_now(),
            "turn": self.turn,
            "role": role,
            "text": text,
            "raw_user_text": text if role == "user" else None,
            "raw_model_output": text if role == "assistant" else None,
            "model": model,
            "runtime": runtime,
            "generation_latency_seconds": latency_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if self.enabled and self._transcript_handle is not None:
            self._transcript_handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )
            self._transcript_handle.flush()

    def save_event(self, reason: str, **fields: object) -> None:
        self.event("SESSION_SAVE", source="session", reason=reason, **fields)

    def finalize(self, state: dict[str, object], *, reason: str) -> None:
        if self._closed:
            return
        elapsed = time.monotonic() - self.started_monotonic
        self.event("SESSION_END", source="session", reason=reason, runtime_seconds=elapsed)
        final = {
            **state,
            "session": self.metadata,
            "event_counts": dict(sorted(self.event_counts.items())),
            "total_turns": self.turn,
            "total_runtime_seconds": elapsed,
            "end_reason": reason,
            "end_timestamp": utc_now(),
        }
        if self.enabled:
            self.final_state_path.write_text(
                json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        for handle in (self._transcript_handle, self._events_handle):
            if handle is not None:
                handle.close()
        self._closed = True


@dataclass(frozen=True)
class CanonicalQueryIntent:
    entity: str | None
    relation_id: int | None
    relation: str | None
    reason: str


@dataclass(frozen=True)
class MutationIntent:
    action: str
    entity: str
    relation_id: int
    value_id: int | None = None
    value: str | None = None
    translator_compatible: bool = True


class CanonicalStateManager:
    """Conservative text-to-existing-canonical-state interface.

    It recognizes only explicit current-state statements and invalidations. It
    never asks the language model to infer state and never extends the fixed
    canonical value vocabulary.
    """

    def __init__(
        self,
        representation: FactorizedStateRepresentation,
        *,
        slots: int = 128,
        store: CanonicalPStore | None = None,
    ) -> None:
        self.representation = representation.cpu().eval()
        self.store = store or CanonicalPStore(CanonicalPConfig(
            slots=slots, width=512, dtype=torch.float32, device="cpu"
        ))
        self.surface_values: dict[int, str] = {}
        self.translator_compatible: dict[int, bool] = {}
        for index in self.store.valid.nonzero(as_tuple=False).flatten().tolist():
            value_id = int(self.store.value_id[index])
            if 0 <= value_id < len(CANONICAL_VALUE_LABELS):
                self.surface_values[index] = CANONICAL_VALUE_LABELS[value_id]
                self.translator_compatible[index] = True

    @staticmethod
    def entity_id(surface: str) -> int:
        return int.from_bytes(
            hashlib.sha256(surface.casefold().encode("utf-8")).digest()[:8], "big"
        ) & ((1 << 63) - 1)

    def vector(self, entity: str, relation_id: int, value_id: int) -> torch.Tensor:
        proof_entities = {"silver key": 0, "gold key": 1}
        factor_entity_id = proof_entities.get(
            entity.casefold(), self.entity_id(entity) % 24,
        )
        with torch.inference_mode():
            return self.representation.encode(
                torch.tensor([factor_entity_id]),
                torch.tensor([relation_id]),
                torch.tensor([value_id]),
                torch.tensor([CANONICAL]),
            )[0].float()

    @staticmethod
    def _clean_entity(value: str) -> str:
        value = re.sub(
            r"^(?:please\s+)?(?:remember|note)(?:\s+that)?\s+",
            "", value, flags=re.IGNORECASE,
        )
        value = re.sub(r"^the\s+", "", value, flags=re.IGNORECASE)
        return value.strip(" \t\n\r.,:;!?\"'")[:160]

    @staticmethod
    def _value_pattern() -> str:
        labels = sorted(CANONICAL_VALUE_LABELS, key=len, reverse=True)
        return "(?:" + "|".join(re.escape(label) for label in labels) + ")"

    @staticmethod
    def _canonical_value(value: str) -> tuple[int, str, bool]:
        cleaned = value.strip(" \t\n\r.,:;!?\"'")[:160]
        known = VALUE_LOOKUP.get(cleaned.casefold())
        if known is not None:
            return known[0], known[1], True
        value_id = int.from_bytes(
            hashlib.sha256(cleaned.casefold().encode("utf-8")).digest()[:8], "big"
        ) % len(CANONICAL_VALUE_LABELS)
        return value_id, cleaned, False

    def extract_mutations(self, text: str) -> list[MutationIntent]:
        value = self._value_pattern()
        flags = re.IGNORECASE
        patterns = (
            (0, rf"^(?P<entity>.+?)\s+(?:currently\s+)?(?:belongs\s+to|is\s+owned\s+by)\s+(?P<value>{value})[.!]?$"),
            (0, rf"^(?P<value>{value})\s+(?:currently\s+)?owns\s+(?P<entity>.+?)[.!]?$"),
            (1, rf"^(?P<entity>.+?)\s+(?:is\s+located\s+in|is\s+located\s+at|is\s+currently\s+in)\s+(?P<value>{value})[.!]?$"),
            (2, rf"^(?:the\s+)?(?:current\s+)?status\s+of\s+(?P<entity>.+?)\s+is\s+(?P<value>{value})[.!]?$"),
            (0, rf"^remember:\s*(?P<entity>.+?)\.owner\s*=\s*(?P<value>{value})\s*$"),
            (1, rf"^remember:\s*(?P<entity>.+?)\.location\s*=\s*(?P<value>{value})\s*$"),
            (2, rf"^remember:\s*(?P<entity>.+?)\.status\s*=\s*(?P<value>{value})\s*$"),
        )
        for relation_id, pattern in patterns:
            match = re.match(pattern, text.strip(), flags)
            if match:
                raw_value = match.group("value")
                value_id, canonical_label = VALUE_LOOKUP[raw_value.casefold()]
                return [MutationIntent(
                    "upsert", self._clean_entity(match.group("entity")),
                    relation_id, value_id, canonical_label, True,
                )]
        free_patterns = (
            (
                1,
                r"^(?:my\s+character|i|we|[\w'-]+)\s+"
                r"(?:left|placed|put|set)\s+(?:the\s+)?(?P<entity>.+?)\s+"
                r"(?:on|in|at|inside|beside|under)\s+(?:the\s+)?(?P<value>[^.!?]+)[.!]?$",
            ),
            (
                0,
                r"^(?:please\s+)?(?:remember|note)(?:\s+that)?\s+"
                r"(?:the\s+)?(?P<entity>.+?)\s+(?:belongs\s+to|is\s+owned\s+by)\s+"
                r"(?P<value>[^.!?]+)[.!]?$",
            ),
            (
                1,
                r"^(?:please\s+)?(?:remember|note)(?:\s+that)?\s+"
                r"(?:the\s+)?(?P<entity>.+?)\s+"
                r"(?:is\s+on|is\s+in|is\s+at|is\s+inside|is\s+beside|is\s+under)\s+"
                r"(?:the\s+)?(?P<value>[^.!?]+)[.!]?$",
            ),
        )
        for relation_id, pattern in free_patterns:
            match = re.match(pattern, text.strip(), flags)
            if match:
                value_id, surface, compatible = self._canonical_value(match.group("value"))
                return [MutationIntent(
                    "upsert", self._clean_entity(match.group("entity")),
                    relation_id, value_id, surface, compatible,
                )]
        invalidate = re.match(
            r"^(?:forget|invalidate):?\s*(?P<entity>.+?)(?:\.|\s+)(?P<relation>owner|location|status)[.!]?$",
            text.strip(), flags,
        )
        if invalidate:
            relation = invalidate.group("relation").casefold()
            return [MutationIntent(
                "invalidate", self._clean_entity(invalidate.group("entity")),
                RELATION_NAMES.index(relation),
            )]
        return []

    def extract_manual_mutations(self, text: str) -> list[MutationIntent]:
        """Keep deterministic explicit overrides ahead of hidden review."""
        stripped = text.strip()
        if not re.match(
            r"^(?:remember:|invalidate:|forget:|please\s+remember\b|note\s+that\b)",
            stripped,
            re.IGNORECASE,
        ):
            return []
        return self.extract_mutations(stripped)

    def infer_query(self, text: str) -> CanonicalQueryIntent:
        lowered = text.casefold()
        labels = [
            str(label) for valid, label in zip(self.store.valid.tolist(), self.store.cache.labels)
            if valid and label and str(label).casefold() in lowered
        ]
        entity = max(labels, key=len) if labels else None
        relation_id = None
        if re.search(r"\b(owner|owns|owned|belongs)\b", lowered):
            relation_id = 0
        elif re.search(r"\b(location|located|where)\b", lowered):
            relation_id = 1
        elif re.search(r"\bstatus\b", lowered):
            relation_id = 2
        if entity is None:
            query_patterns = (
                r"\bwho\s+(?:owns|owned)\s+(?:the\s+)?(?P<entity>[^?.!]+)",
                r"\b(?:where\s+is|location\s+of)\s+(?:the\s+)?(?P<entity>[^?.!]+)",
                r"\bwhere\s+did\s+(?:i|we|my\s+character)\s+(?:leave|put|place|set)\s+"
                r"(?:the\s+)?(?P<entity>[^?.!]+)",
                r"\bstatus\s+of\s+(?:the\s+)?(?P<entity>[^?.!]+)",
            )
            for pattern in query_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entity = self._clean_entity(match.group("entity"))
                    break
        if entity is not None and not labels:
            words = set(re.findall(r"[\w'-]+", entity.casefold()))
            aliases = []
            for valid, label in zip(self.store.valid.tolist(), self.store.cache.labels):
                if not valid or not label:
                    continue
                label_words = set(re.findall(r"[\w'-]+", str(label).casefold()))
                if words and words <= label_words:
                    aliases.append(str(label))
            if len(set(aliases)) == 1:
                entity = aliases[0]
        if entity is None:
            return CanonicalQueryIntent(None, relation_id, None, "no known entity mentioned")
        if relation_id is None:
            relations = {
                int(self.store.relation_id[index])
                for index in self.store.valid.nonzero(as_tuple=False).flatten().tolist()
                if self.store.cache.labels[index]
                and str(self.store.cache.labels[index]).casefold() == entity.casefold()
            }
            if len(relations) == 1:
                relation_id = next(iter(relations))
        relation = None if relation_id is None else RELATION_NAMES[relation_id]
        return CanonicalQueryIntent(entity, relation_id, relation, "explicit entity surface match")

    def _matching_slots(self, entity: str, relation_id: int) -> list[int]:
        return [
            index for index in self.store.valid.nonzero(as_tuple=False).flatten().tolist()
            if self.store.cache.labels[index]
            and str(self.store.cache.labels[index]).casefold() == entity.casefold()
            and int(self.store.relation_id[index]) == relation_id
        ]

    def apply(self, intents: Iterable[MutationIntent], recorder: SessionRecorder) -> None:
        for intent in intents:
            matches = self._matching_slots(intent.entity, intent.relation_id)
            if intent.action == "invalidate":
                if not matches:
                    recorder.event(
                        "P_IGNORE", source="p_cache", entity=intent.entity,
                        relation=RELATION_NAMES[intent.relation_id], reason="no active state to invalidate",
                    )
                    continue
                for slot in matches:
                    before = self.entry(slot)
                    self.store.invalidate(slot)
                    self.surface_values.pop(slot, None)
                    self.translator_compatible.pop(slot, None)
                    recorder.event("P_INVALIDATE", source="p_cache", slot_id=slot, before=before)
                continue
            assert intent.value_id is not None and intent.value is not None
            vector = self.vector(intent.entity, intent.relation_id, intent.value_id)
            if matches:
                slot = matches[0]
                existing_surface = self.surface_values.get(slot, "").casefold()
                if (
                    int(self.store.value_id[slot]) == intent.value_id
                    and existing_surface == intent.value.casefold()
                ):
                    self.store.cache.keep(slot)
                    recorder.event(
                        "P_KEEP", source="p_cache", slot_id=slot,
                        entity=intent.entity, relation=RELATION_NAMES[intent.relation_id],
                        value=intent.value,
                    )
                else:
                    before = self.entry(slot)
                    self.store.modify(
                        slot, vector, entity_id=self.entity_id(intent.entity),
                        relation_id=intent.relation_id, value_id=intent.value_id,
                        metadata_id=CANONICAL, confidence=1.0,
                        source=SlotSource.CORRECTION,
                    )
                    self.surface_values[slot] = intent.value
                    self.translator_compatible[slot] = intent.translator_compatible
                    recorder.event(
                        "P_MODIFY", source="p_cache", slot_id=slot, before=before,
                        after=self.entry(slot),
                    )
                self.surface_values[slot] = intent.value
                self.translator_compatible[slot] = intent.translator_compatible
                continue
            slot, operation = self.store.create(
                vector, entity_id=self.entity_id(intent.entity),
                relation_id=intent.relation_id, value_id=intent.value_id,
                metadata_id=CANONICAL, slot_type=SlotType.FACT,
                confidence=1.0, importance=0.7, freshness=Freshness.FRESH,
                persistence=Persistence.SESSION, source=SlotSource.CONVERSATION,
                label=intent.entity,
            )
            event = {
                StateOperation.CREATE: "P_CREATE",
                StateOperation.MERGE: "P_MERGE",
                StateOperation.IGNORE: "P_IGNORE",
            }[operation]
            if slot >= 0:
                self.surface_values[slot] = intent.value
                self.translator_compatible[slot] = intent.translator_compatible
            recorder.event(
                event, source="p_cache", slot_id=None if slot < 0 else slot,
                entity=intent.entity, relation=RELATION_NAMES[intent.relation_id],
                value=intent.value,
                translator_compatible=intent.translator_compatible,
            )

    def entry(self, index: int) -> dict[str, object]:
        value_id = int(self.store.value_id[index])
        return {
            "slot_id": index,
            "entity": self.store.cache.labels[index],
            "entity_id": int(self.store.entity_id[index]),
            "relation": RELATION_NAMES[int(self.store.relation_id[index])],
            "relation_id": int(self.store.relation_id[index]),
            "value": self.surface_values.get(
                index,
                CANONICAL_VALUE_LABELS[value_id]
                if 0 <= value_id < len(CANONICAL_VALUE_LABELS) else None,
            ),
            "value_id": value_id,
            "translator_compatible": self.translator_compatible.get(index, True),
            "metadata_id": int(self.store.canonical_metadata_id[index]),
            "confidence": float(self.store.cache.confidence[index]),
            "importance": float(self.store.cache.importance[index]),
            "freshness": Freshness(int(self.store.cache.freshness[index])).name.lower(),
            "persistence": Persistence(int(self.store.cache.persistence[index])).name.lower(),
            "source": SlotSource(int(self.store.cache.source[index])).name.lower(),
            "last_updated": int(self.store.cache.last_updated[index]),
        }

    def snapshot(self) -> list[dict[str, object]]:
        return [
            self.entry(index)
            for index in self.store.valid.nonzero(as_tuple=False).flatten().tolist()
        ]

    def translation_store(self, *, include_open_values: bool = False) -> CanonicalPStore:
        compatible = [
            index for index in self.store.valid.nonzero(as_tuple=False).flatten().tolist()
            if include_open_values or self.translator_compatible.get(index, True)
        ]
        result = CanonicalPStore(CanonicalPConfig(
            slots=max(1, len(compatible)), width=512, dtype=torch.float32,
            device="cpu", merge_similarity=1.0,
        ))
        value_surfaces = {}
        for index in compatible:
            new_slot, _operation = result.create(
                self.store.canonical_values[index],
                entity_id=int(self.store.entity_id[index]),
                relation_id=int(self.store.relation_id[index]),
                value_id=int(self.store.value_id[index]),
                metadata_id=int(self.store.canonical_metadata_id[index]),
                slot_type=SlotType(int(self.store.cache.slot_type[index])),
                confidence=float(self.store.cache.confidence[index]),
                importance=float(self.store.cache.importance[index]),
                freshness=Freshness(int(self.store.cache.freshness[index])),
                persistence=Persistence(int(self.store.cache.persistence[index])),
                source=SlotSource(int(self.store.cache.source[index])),
                label=self.store.cache.labels[index],
            )
            if new_slot >= 0:
                value_id = int(self.store.value_id[index])
                value_surfaces[new_slot] = self.surface_values.get(
                    index,
                    CANONICAL_VALUE_LABELS[value_id]
                    if 0 <= value_id < len(CANONICAL_VALUE_LABELS) else None,
                )
        # Runtime-only canonical surface metadata.  Canonical snapshots remain
        # unchanged and never store model token identifiers or hidden vectors.
        result._pcm_value_surfaces = value_surfaces
        return result

    def save_runtime_metadata(self, path: str | Path) -> None:
        payload = {
            "format": "pcm-interactive-p-metadata-v1",
            "surface_values": {str(key): value for key, value in self.surface_values.items()},
            "translator_compatible": {
                str(key): value for key, value in self.translator_compatible.items()
            },
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    def load_runtime_metadata(self, path: str | Path) -> None:
        source = Path(path)
        if not source.is_file():
            return
        payload = json.loads(source.read_text())
        if payload.get("format") != "pcm-interactive-p-metadata-v1":
            raise ValueError("unsupported interactive P metadata")
        self.surface_values.update({
            int(key): str(value) for key, value in payload["surface_values"].items()
        })
        self.translator_compatible.update({
            int(key): bool(value)
            for key, value in payload["translator_compatible"].items()
        })


class PersonalityManager:
    """One reusable `.ppkg` connection with explicit evidence extraction."""

    def __init__(
        self,
        path: str | Path,
        representation: FactorizedStateRepresentation,
        *,
        create: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            if not create:
                raise FileNotFoundError(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            created = PersonalityPackage.create(
                self.path, package_id=f"planner-personality-{safe_session_stamp(utc_now())}"
            )
            created.close()
        self.package = PersonalityPackage(self.path, validate=True)
        self.router = PersonalityRouter()
        self.canonicalizer = FactorizedPersonalityCanonicalizer(
            representation, value_labels=CANONICAL_VALUE_LABELS,
        )
        self.last_selection = None
        self.mutations: list[dict[str, object]] = []

    @staticmethod
    def context(text: str) -> tuple[str, str]:
        lowered = text.casefold()
        if any(word in lowered for word in ("code", "debug", "error", "python", "cuda")):
            return "technical", "debugging"
        if any(word in lowered for word in ("story", "poem", "creative", "character")):
            return "creative", "writing"
        if any(word in lowered for word in ("roleplay", " rp ", "scene", "dialogue")):
            return "roleplay", "roleplay"
        return "chat", "general"

    def extract_evidence(self, text: str, *, turn: int, timestamp: str) -> EvidenceRecord | None:
        patterns = (
            r"^i\s+(?:strongly\s+)?prefer\s+(?P<value>concise|detailed|direct|structured|expressive)(?:\s+(?:responses|replies|answers))?[.!]?$",
            r"^please\s+(?:always\s+)?(?:be|respond\s+in\s+a)\s+(?P<value>concise|detailed|direct|structured|expressive)(?:\s+(?:style|way))?[.!]?$",
        )
        for pattern in patterns:
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                interaction, domain = self.context(text)
                return EvidenceRecord(
                    id=(
                        f"session-turn-{turn}-"
                        f"{hashlib.sha256((timestamp + text).encode()).hexdigest()[:16]}"
                    ),
                    entry_type=PersonalityType.INTERACTION_STYLE.value,
                    subject="user", relation="response_style",
                    value=match.group("value").casefold(), context=domain,
                    scope=interaction, confidence=0.95,
                    source_authority=EvidenceAuthority.EXPLICIT_USER.value,
                    timestamp=timestamp,
                    archive_reference=f"session://turn/{turn}/user",
                )
        return None

    def ingest(self, record: EvidenceRecord, recorder: SessionRecorder) -> None:
        change_count = len(self.package.changes())
        decision = self.package.ingest(record)
        payload = asdict(decision)
        changes = self.package.changes()[change_count:]
        self.mutations.append(payload)
        recorder.event(
            "PPKG_UPDATE", source="ppkg", evidence=asdict(record),
            decision=payload, changes=changes,
        )
        if decision.promoted:
            recorder.event("PPKG_PROMOTION", source="ppkg", **payload)
        contradiction_changes = [
            change for change in changes
            if change["action"] in {"lower_confidence", "contradict", "supersede"}
        ]
        if contradiction_changes:
            recorder.event(
                "PPKG_CONTRADICTION", source="ppkg", decision=payload,
                changes=contradiction_changes,
            )

    def query(
        self,
        text: str,
        recorder: SessionRecorder,
        *,
        top_k: int = 4,
    ) -> CanonicalPStore | None:
        interaction, domain = self.context(text)
        query = PersonalityQuery(
            subject="user", interaction_type=interaction, domain=domain,
            relation="response_style", timestamp=utc_now(),
        )
        recorder.event("PPKG_QUERY", source="ppkg", query=asdict(query), top_k=top_k)
        selection = self.router.retrieve(self.package, query, top_k=top_k)
        self.last_selection = selection
        recorder.event(
            "PPKG_CANDIDATES", source="ppkg",
            candidate_count=selection.route.candidate_count,
            accepted=selection.route.accepted,
            entry_ids=list(selection.route.entry_ids), scores=list(selection.route.scores),
            header_bytes_read=selection.route.header_bytes_read,
        )
        if not selection.entries:
            return None
        recorder.event(
            "PPKG_LOAD", source="ppkg",
            entries=[asdict(entry) for entry in selection.entries],
            entry_bytes_read=selection.entry_bytes_read,
            latency_seconds=selection.retrieval_latency_seconds,
            translator_compatible_entries=[
                entry.id for entry in selection.entries
                if entry.value.casefold() in VALUE_LOOKUP
            ],
        )
        compatible_entries = tuple(
            entry for entry in selection.entries
            if entry.value.casefold() in VALUE_LOOKUP
        )
        if not compatible_entries:
            return None
        store = CanonicalPStore(CanonicalPConfig(
            slots=len(compatible_entries), width=512, dtype=torch.float32,
            device="cpu", merge_similarity=1.0,
        ))
        for entry in compatible_entries:
            vector, ids = self.canonicalizer.encode(entry)
            store.create(
                vector, entity_id=ids[0], relation_id=ids[1], value_id=ids[2],
                metadata_id=ids[3], slot_type=SlotType.FACT,
                confidence=entry.confidence, importance=entry.importance,
                freshness=Freshness.FRESH, persistence=Persistence.DURABLE,
                source=SlotSource.CONVERSATION, label=entry.subject,
            )
        return store

    def visible_entries(
        self, *, limit: int | None = None, offset: int = 0,
    ) -> list[dict[str, object]]:
        return [
            asdict(entry)
            for entry in self.package.entries(
                status=PersonalityStatus.ACTIVE.value,
                limit=limit,
                offset=offset,
            )
        ]

    def visible_entry_page(
        self, *, limit: int = 100, offset: int = 0,
    ) -> dict[str, object]:
        """Return one bounded, deterministic debug page without loading the package."""
        if limit <= 0 or limit > 200:
            raise ValueError("personality debug limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("personality debug offset cannot be negative")
        total = self.package.entry_count(status=PersonalityStatus.ACTIVE.value)
        entries = self.visible_entries(limit=limit, offset=offset)
        returned = len(entries)
        return {
            "entries": entries,
            "total_active": total,
            "returned": returned,
            "limit": limit,
            "offset": offset,
            "truncated": offset + returned < total,
        }

    def checkpoint(self) -> str:
        return self.package.checkpoint(updated_at=utc_now())

    def close(self) -> None:
        self.package.close(checkpoint=True)
