"""Disk-resident, model-independent durable personality memory.

The package is deliberately mechanical: evidence is accumulated on CPU/disk,
promotion is deterministic, and only selected canonical entries are activated
through the existing P translation boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Iterable, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.cache import Freshness, Persistence, SlotSource, SlotType
from pcm.planner.representation import CANONICAL, FactorizedStateRepresentation
from pcm.planner.split_translator import ByteEntityEncoder
from pcm.planner.canonical import CANONICAL_P_PROTOCOL


PPKG_FORMAT = "pcm-personality-package-v1"
PPKG_PROTOCOL = "pcm-canonical-personality-v1"


class PersonalityType(str, Enum):
    TRAIT = "trait"
    PREFERENCE = "preference"
    RELATIONSHIP_PATTERN = "relationship_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    INTERACTION_STYLE = "interaction_style"
    TERMINOLOGY = "terminology"
    HABIT = "habit"
    RESPONSE_TENDENCY = "response_tendency"
    CONTEXTUAL_TENDENCY = "contextual_tendency"


class EvidenceAuthority(str, Enum):
    EXPLICIT_USER = "explicit_user_correction_or_statement"
    EXTERNALLY_VERIFIED = "externally_verified_observation"
    REPEATED_OBSERVED = "repeated_observed_interaction_behavior"
    SINGLE_OBSERVED = "single_observed_behavior"
    MODEL_INFERENCE = "model_inference"
    MODEL_UNSUPPORTED = "model_generated_unsupported_claim"

    @property
    def weight(self) -> float:
        return {
            EvidenceAuthority.EXPLICIT_USER: 1.0,
            EvidenceAuthority.EXTERNALLY_VERIFIED: 0.9,
            EvidenceAuthority.REPEATED_OBSERVED: 0.7,
            EvidenceAuthority.SINGLE_OBSERVED: 0.45,
            EvidenceAuthority.MODEL_INFERENCE: 0.15,
            EvidenceAuthority.MODEL_UNSUPPORTED: 0.0,
        }[self]


class PersonalityStatus(str, Enum):
    ACTIVE = "active"
    CONTRADICTED = "contradicted"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    entry_type: str
    subject: str
    relation: str
    value: str
    context: str
    scope: str
    confidence: float
    source_authority: str
    timestamp: str
    archive_reference: str
    polarity: int = 1
    relationship: str | None = None
    connected_evidence_ids: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.subject or not self.relation or not self.value:
            raise ValueError("evidence id, subject, relation, and value are required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("evidence confidence must be in [0, 1]")
        if self.polarity not in (-1, 1):
            raise ValueError("evidence polarity must be -1 or 1")
        EvidenceAuthority(self.source_authority)
        if not self.archive_reference:
            raise ValueError("evidence must reference an archive/source event")


@dataclass(frozen=True)
class PersonalityEntry:
    id: str
    entry_type: str
    subject: str
    relation: str
    value: str
    scope: str
    relationship: str | None
    strength: float
    confidence: float
    importance: float
    evidence_count: int
    context_diversity: int
    last_reinforced: str
    created_at: str
    updated_at: str
    source_authority: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    status: str = PersonalityStatus.ACTIVE.value
    extension: dict[str, object] | None = None


@dataclass(frozen=True)
class PromotionPolicy:
    """Public coefficients for the conservative v1 promotion score."""

    promotion_threshold: float = 1.25
    context_diversity_coefficient: float = 0.8
    connectivity_coefficient: float = 0.25
    contradiction_weight: float = 0.75
    confidence_prior: float = 0.5
    explicit_override_confidence: float = 0.85

    def score(self, evidence: Sequence[EvidenceRecord]) -> dict[str, float]:
        supporting = [record for record in evidence if record.polarity > 0]
        weighted_support = sum(
            record.confidence * EvidenceAuthority(record.source_authority).weight
            for record in supporting
        )
        diversity = len({record.context for record in supporting})
        linked = len({
            linked_id
            for record in supporting
            for linked_id in record.connected_evidence_ids
        })
        diversity_factor = 1 + self.context_diversity_coefficient * math.log1p(
            max(0, diversity - 1)
        )
        connectivity_factor = 1 + self.connectivity_coefficient * math.log1p(linked)
        promotion_score = weighted_support * diversity_factor * connectivity_factor
        return {
            "weighted_support": weighted_support,
            "context_diversity": float(diversity),
            "connected_evidence": float(linked),
            "diversity_factor": diversity_factor,
            "connectivity_factor": connectivity_factor,
            "promotion_score": promotion_score,
        }


@dataclass(frozen=True)
class PromotionDecision:
    evidence_id: str
    promoted: bool
    entry_id: str | None
    action: str
    promotion_score: float
    threshold: float
    reason: str


@dataclass(frozen=True)
class PersonalityQuery:
    subject: str
    interaction_type: str
    domain: str
    relationship: str | None = None
    relation: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class PersonalityRoute:
    entry_ids: tuple[str, ...]
    scores: tuple[float, ...]
    candidate_count: int
    header_bytes_read: int
    accepted: int


@dataclass(frozen=True)
class PersonalitySelection:
    entries: tuple[PersonalityEntry, ...]
    route: PersonalityRoute
    entry_bytes_read: int
    retrieval_latency_seconds: float

    @property
    def logical_bytes_read(self) -> int:
        return self.route.header_bytes_read + self.entry_bytes_read


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _stable_id(prefix: str, *fields: object) -> str:
    digest = hashlib.sha256(_canonical_json(fields).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _entry_to_row(entry: PersonalityEntry) -> tuple[object, ...]:
    return (
        entry.id, entry.entry_type, entry.subject, entry.relation, entry.value,
        entry.scope, entry.relationship, entry.strength, entry.confidence,
        entry.importance, entry.evidence_count, entry.context_diversity,
        entry.last_reinforced, entry.created_at, entry.updated_at,
        entry.source_authority, _canonical_json(entry.supporting_evidence_ids),
        _canonical_json(entry.contradicting_evidence_ids), entry.status,
        _canonical_json(entry.extension or {}),
    )


def _row_to_entry(row: Sequence[object]) -> PersonalityEntry:
    return PersonalityEntry(
        id=str(row[0]), entry_type=str(row[1]), subject=str(row[2]),
        relation=str(row[3]), value=str(row[4]), scope=str(row[5]),
        relationship=None if row[6] is None else str(row[6]),
        strength=float(row[7]), confidence=float(row[8]), importance=float(row[9]),
        evidence_count=int(row[10]), context_diversity=int(row[11]),
        last_reinforced=str(row[12]), created_at=str(row[13]), updated_at=str(row[14]),
        source_authority=str(row[15]),
        supporting_evidence_ids=tuple(json.loads(str(row[16]))),
        contradicting_evidence_ids=tuple(json.loads(str(row[17]))),
        status=str(row[18]), extension=json.loads(str(row[19])),
    )


ENTRY_COLUMNS = (
    "id,type,subject,relation,value,scope,relationship,strength,confidence,importance,"
    "evidence_count,context_diversity,last_reinforced,created_at,updated_at,"
    "source_authority,supporting_ids,contradicting_ids,status,extension_json"
)


class PersonalityPackage:
    """A SQLite-backed `.ppkg`; opening it allocates no CUDA tensors."""

    def __init__(self, path: str | Path, *, validate: bool = True) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        # Web sessions create the package on the launcher thread and execute
        # serialized turns on the HTTP worker thread. SQLite's transactional
        # behavior is unchanged. Cross-thread access is enabled only so the
        # gateway's single session lock can own that serialization boundary.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only = ON")
        self._closed = False
        self._validate_header()
        self._dirty = self.metadata().get("integrity_state", "clean") != "clean"
        if validate:
            self.validate_checksum()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        package_id: str,
        created_at: str | None = None,
        metadata: dict[str, object] | None = None,
        overwrite: bool = False,
    ) -> "PersonalityPackage":
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        if path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            PRAGMA page_size = 4096;
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE entries (
              id TEXT PRIMARY KEY, type TEXT NOT NULL, subject TEXT NOT NULL,
              relation TEXT NOT NULL, value TEXT NOT NULL, scope TEXT NOT NULL,
              relationship TEXT, strength REAL NOT NULL, confidence REAL NOT NULL,
              importance REAL NOT NULL, evidence_count INTEGER NOT NULL,
              context_diversity INTEGER NOT NULL, last_reinforced TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              source_authority TEXT NOT NULL, supporting_ids TEXT NOT NULL,
              contradicting_ids TEXT NOT NULL, status TEXT NOT NULL,
              extension_json TEXT NOT NULL
            );
            CREATE INDEX entry_lookup ON entries(status, subject, relation, scope, relationship);
            CREATE INDEX entry_subject_route ON entries(
              status,subject,importance DESC,confidence DESC,id
            );
            CREATE INDEX entry_scope_route ON entries(
              status,scope,importance DESC,confidence DESC,id
            );
            CREATE INDEX entry_relationship_route ON entries(
              status,relationship,importance DESC,confidence DESC,id
            );
            CREATE TABLE evidence (
              id TEXT PRIMARY KEY, entry_type TEXT NOT NULL, subject TEXT NOT NULL,
              relation TEXT NOT NULL, value TEXT NOT NULL, context TEXT NOT NULL,
              scope TEXT NOT NULL, confidence REAL NOT NULL,
              source_authority TEXT NOT NULL, timestamp TEXT NOT NULL,
              archive_reference TEXT NOT NULL, polarity INTEGER NOT NULL,
              relationship TEXT, connected_ids TEXT NOT NULL, note TEXT
            );
            CREATE INDEX evidence_candidate ON evidence(entry_type, subject, relation, scope, relationship, value);
            CREATE TABLE changes (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT NOT NULL,
              timestamp TEXT NOT NULL, action TEXT NOT NULL, entry_id TEXT NOT NULL,
              before_json TEXT, after_json TEXT
            );
            """
        )
        stamp = created_at or _utc_now()
        values = {
            "format": PPKG_FORMAT,
            "protocol": PPKG_PROTOCOL,
            "canonical_p_protocol": CANONICAL_P_PROTOCOL,
            "package_id": package_id,
            "created_at": stamp,
            "updated_at": stamp,
            "schema_version": "1",
            "extensions": _canonical_json(metadata or {}),
            "integrity_state": "clean",
            "content_sha256": "pending",
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", sorted(values.items())
        )
        connection.commit()
        connection.close()
        package = cls(path, validate=False)
        package._dirty = True
        package.checkpoint(updated_at=stamp)
        return package

    def close(self, *, checkpoint: bool = True) -> None:
        if self._closed:
            return
        if checkpoint and self._dirty:
            self.checkpoint()
        self._connection.close()
        self._closed = True

    def __enter__(self) -> "PersonalityPackage":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def package_id(self) -> str:
        return self.metadata()["package_id"]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def metadata(self) -> dict[str, str]:
        return {
            str(row[0]): str(row[1])
            for row in self._connection.execute("SELECT key,value FROM metadata")
        }

    def _validate_header(self) -> None:
        try:
            metadata = self.metadata()
        except sqlite3.DatabaseError as error:
            raise ValueError("corrupt personality package") from error
        if metadata.get("format") != PPKG_FORMAT:
            raise ValueError("unsupported personality package format")
        if metadata.get("protocol") != PPKG_PROTOCOL:
            raise ValueError("incompatible personality protocol")
        if metadata.get("canonical_p_protocol") != CANONICAL_P_PROTOCOL:
            raise ValueError("incompatible canonical P protocol")
        if metadata.get("schema_version") != "1":
            raise ValueError("unsupported personality package schema")

    def _semantic_checksum(self) -> str:
        digest = hashlib.sha256()
        metadata = [
            tuple(row)
            for row in self._connection.execute(
                "SELECT key,value FROM metadata WHERE key != 'content_sha256' ORDER BY key"
            )
        ]
        digest.update(_canonical_json(metadata).encode())
        for table, order in (
            ("entries", "id"), ("evidence", "id"), ("changes", "sequence")
        ):
            for row in self._connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                digest.update(_canonical_json(tuple(row)).encode())
        return digest.hexdigest()

    def validate_checksum(self) -> None:
        if self._dirty or self.metadata().get("integrity_state", "clean") != "clean":
            raise ValueError("personality package has uncheckpointed changes")
        expected = self.metadata().get("content_sha256")
        try:
            actual = self._semantic_checksum()
        except sqlite3.DatabaseError as error:
            raise ValueError("corrupt personality package") from error
        if not expected or expected != actual:
            raise ValueError("personality package checksum does not match")

    def verify(self) -> None:
        """Explicit full-package integrity boundary."""
        self.validate_checksum()

    def _writable(self) -> None:
        self._connection.execute("PRAGMA query_only = OFF")

    def _readonly(self) -> None:
        self._connection.execute("PRAGMA query_only = ON")

    def _mark_dirty(self, updated_at: str) -> None:
        self._connection.execute(
            "UPDATE metadata SET value=? WHERE key='updated_at'", (updated_at,)
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('integrity_state','dirty')"
        )
        self._dirty = True

    def checkpoint(self, *, updated_at: str | None = None) -> str:
        """Atomically seal all committed mutations with one semantic checksum."""
        if self._closed:
            raise RuntimeError("personality package is closed")
        if not self._dirty:
            return self.metadata()["content_sha256"]
        self._writable()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "UPDATE metadata SET value=? WHERE key='updated_at'",
                (updated_at or self.metadata()["updated_at"],),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES ('integrity_state','clean')"
            )
            self._connection.execute(
                "UPDATE metadata SET value='pending' WHERE key='content_sha256'"
            )
            checksum = self._semantic_checksum()
            self._connection.execute(
                "UPDATE metadata SET value=? WHERE key='content_sha256'", (checksum,)
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            self._readonly()
            raise
        self._dirty = False
        self._readonly()
        return checksum

    def export(self, path: str | Path) -> Path:
        """Checkpoint, snapshot with SQLite backup, and verify the snapshot."""
        self.checkpoint()
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        target = sqlite3.connect(destination)
        try:
            self._connection.backup(target)
        finally:
            target.close()
        with PersonalityPackage(destination, validate=True):
            pass
        return destination

    def _evidence_rows(self, record: EvidenceRecord) -> None:
        self._connection.execute(
            """INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.id, record.entry_type, record.subject, record.relation,
                record.value, record.context, record.scope, record.confidence,
                record.source_authority, record.timestamp, record.archive_reference,
                record.polarity, record.relationship,
                _canonical_json(record.connected_evidence_ids), record.note,
            ),
        )

    def evidence(self, evidence_id: str) -> EvidenceRecord:
        row = self._connection.execute(
            "SELECT * FROM evidence WHERE id=?", (evidence_id,)
        ).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        return EvidenceRecord(
            id=row[0], entry_type=row[1], subject=row[2], relation=row[3], value=row[4],
            context=row[5], scope=row[6], confidence=row[7], source_authority=row[8],
            timestamp=row[9], archive_reference=row[10], polarity=row[11],
            relationship=row[12], connected_evidence_ids=tuple(json.loads(row[13])),
            note=row[14],
        )

    def _candidate_records(self, record: EvidenceRecord) -> list[EvidenceRecord]:
        rows = self._connection.execute(
            """SELECT id FROM evidence WHERE entry_type=? AND subject=? AND relation=?
               AND scope=? AND relationship IS ? ORDER BY id""",
            (record.entry_type, record.subject, record.relation, record.scope, record.relationship),
        )
        return [self.evidence(str(row[0])) for row in rows]

    def entry_count(self, *, status: str | None = None) -> int:
        if status is None:
            row = self._connection.execute("SELECT COUNT(*) FROM entries").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM entries WHERE status=?", (status,)
            ).fetchone()
        assert row is not None
        return int(row[0])

    def entries(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[PersonalityEntry]:
        if limit is not None and limit <= 0:
            raise ValueError("entry limit must be positive")
        if offset < 0:
            raise ValueError("entry offset cannot be negative")
        where = "" if status is None else " WHERE status=?"
        parameters: list[object] = [] if status is None else [status]
        pagination = ""
        if limit is not None:
            pagination = " LIMIT ? OFFSET ?"
            parameters.extend((limit, offset))
        elif offset:
            pagination = " LIMIT -1 OFFSET ?"
            parameters.append(offset)
        rows = self._connection.execute(
            f"SELECT {ENTRY_COLUMNS} FROM entries{where} ORDER BY id{pagination}",
            parameters,
        )
        return [_row_to_entry(tuple(row)) for row in rows]

    def entry(self, entry_id: str) -> PersonalityEntry:
        row = self._connection.execute(
            f"SELECT {ENTRY_COLUMNS} FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        if row is None:
            raise KeyError(entry_id)
        return _row_to_entry(tuple(row))

    def routing_headers(
        self,
        *,
        subject: str,
        scopes: Sequence[str],
        relationship: str | None,
        candidate_limit: int,
    ) -> list[sqlite3.Row]:
        """Indexed/coarse prefilter returning bounded canonical header rows."""
        if candidate_limit <= 0:
            raise ValueError("candidate limit must be positive")
        columns = (
            "id,type,subject,relation,value,scope,relationship,strength,"
            "confidence,importance,updated_at"
        )
        per_bucket = max(32, candidate_limit // 4)
        available_indexes = {
            str(row[0]) for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        subject_hint = (
            "INDEXED BY entry_subject_route"
            if "entry_subject_route" in available_indexes else ""
        )
        scope_hint = (
            "INDEXED BY entry_scope_route"
            if "entry_scope_route" in available_indexes else ""
        )
        relationship_hint = (
            "INDEXED BY entry_relationship_route"
            if "entry_relationship_route" in available_indexes else ""
        )
        statements: list[tuple[str, tuple[object, ...]]] = [(
            f"""SELECT {columns} FROM entries {subject_hint}
                WHERE status=? AND subject=?
                ORDER BY importance DESC,confidence DESC,id LIMIT ?""",
            (PersonalityStatus.ACTIVE.value, subject, per_bucket),
        )]
        for scope in dict.fromkeys((*scopes, "global")):
            statements.append((
                f"""SELECT {columns} FROM entries {scope_hint}
                    WHERE status=? AND scope=?
                    ORDER BY importance DESC,confidence DESC,id LIMIT ?""",
                (PersonalityStatus.ACTIVE.value, scope, per_bucket),
            ))
        if relationship is not None:
            statements.append((
                f"""SELECT {columns} FROM entries {relationship_hint}
                    WHERE status=? AND relationship=?
                    ORDER BY importance DESC,confidence DESC,id LIMIT ?""",
                (PersonalityStatus.ACTIVE.value, relationship, per_bucket),
            ))
        rows: dict[str, sqlite3.Row] = {}
        for statement, parameters in statements:
            for row in self._connection.execute(statement, parameters):
                rows.setdefault(str(row[0]), row)
        return list(rows.values())

    def _entry_json(self, entry: PersonalityEntry | None) -> str | None:
        return None if entry is None else _canonical_json(asdict(entry))

    def _upsert_entry(
        self,
        entry: PersonalityEntry,
        *,
        transaction_id: str,
        action: str,
        previous: PersonalityEntry | None,
    ) -> None:
        self._connection.execute(
            f"INSERT OR REPLACE INTO entries({ENTRY_COLUMNS}) VALUES ({','.join('?' for _ in range(20))})",
            _entry_to_row(entry),
        )
        self._connection.execute(
            """INSERT INTO changes(transaction_id,timestamp,action,entry_id,before_json,after_json)
               VALUES (?,?,?,?,?,?)""",
            (
                transaction_id, entry.updated_at, action, entry.id,
                self._entry_json(previous), self._entry_json(entry),
            ),
        )

    def ingest(
        self,
        record: EvidenceRecord,
        *,
        policy: PromotionPolicy = PromotionPolicy(),
        importance: float = 0.5,
    ) -> PromotionDecision:
        """Persist evidence, then deterministically recompute its candidate family."""
        self._writable()
        try:
            self._evidence_rows(record)
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            self._readonly()
            raise ValueError(f"duplicate evidence id: {record.id}") from error
        family = self._candidate_records(record)
        support = [item for item in family if item.value == record.value and item.polarity > 0]
        direct_negative = [
            item for item in family if item.value == record.value and item.polarity < 0
        ]
        opposing = [
            item for item in family if item.value != record.value and item.polarity > 0
        ] + direct_negative
        score = policy.score(support)
        opposing_weight = sum(
            item.confidence * EvidenceAuthority(item.source_authority).weight
            for item in opposing
        )
        net_score = max(0.0, score["promotion_score"] - policy.contradiction_weight * opposing_weight)
        authorities = {EvidenceAuthority(item.source_authority) for item in support}
        explicit_override = any(
            EvidenceAuthority(item.source_authority) == EvidenceAuthority.EXPLICIT_USER
            and item.confidence >= policy.explicit_override_confidence
            for item in support
        )
        inference_only = bool(authorities) and authorities <= {
            EvidenceAuthority.MODEL_INFERENCE, EvidenceAuthority.MODEL_UNSUPPORTED
        }
        promotable = (
            bool(support)
            and not inference_only
            and (
                explicit_override
                or (len(support) >= 3 and net_score >= policy.promotion_threshold)
            )
        )
        entry_id = _stable_id(
            "personality", record.entry_type, record.subject, record.relation,
            record.value, record.scope, record.relationship,
        )
        transaction_id = _stable_id("change", record.id, record.timestamp)
        existing_row = self._connection.execute(
            f"SELECT {ENTRY_COLUMNS} FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        existing = None if existing_row is None else _row_to_entry(tuple(existing_row))
        action = "evidence_only"
        reason = "promotion threshold not reached"
        promoted_entry_id: str | None = None

        if promotable:
            weighted_support = score["weighted_support"]
            confidence = weighted_support / (
                weighted_support + opposing_weight + policy.confidence_prior
            )
            if explicit_override:
                confidence = max(confidence, 0.9)
            strength = min(1.0, net_score / (2 * policy.promotion_threshold))
            strongest = max(
                support,
                key=lambda item: (
                    EvidenceAuthority(item.source_authority).weight, item.confidence
                ),
            )
            created_at = existing.created_at if existing else record.timestamp
            entry = PersonalityEntry(
                id=entry_id, entry_type=record.entry_type, subject=record.subject,
                relation=record.relation, value=record.value, scope=record.scope,
                relationship=record.relationship, strength=strength,
                confidence=confidence, importance=importance,
                evidence_count=len(support),
                context_diversity=int(score["context_diversity"]),
                last_reinforced=max(item.timestamp for item in support),
                created_at=created_at, updated_at=record.timestamp,
                source_authority=strongest.source_authority,
                supporting_evidence_ids=tuple(sorted(item.id for item in support)),
                contradicting_evidence_ids=tuple(sorted(item.id for item in opposing)),
                status=PersonalityStatus.ACTIVE.value,
                extension=(existing.extension if existing else {}),
            )
            # A correction never erases the old conclusion; it supersedes it.
            active_opponents = self._connection.execute(
                f"""SELECT {ENTRY_COLUMNS} FROM entries WHERE type=? AND subject=?
                    AND relation=? AND scope=? AND relationship IS ? AND status=? AND id != ?""",
                (
                    record.entry_type, record.subject, record.relation, record.scope,
                    record.relationship, PersonalityStatus.ACTIVE.value, entry_id,
                ),
            ).fetchall()
            for opponent_row in active_opponents:
                opponent = _row_to_entry(tuple(opponent_row))
                new_status = (
                    PersonalityStatus.SUPERSEDED.value
                    if explicit_override else PersonalityStatus.CONTRADICTED.value
                )
                changed = replace(
                    opponent,
                    confidence=max(0.0, opponent.confidence - opposing_weight / (1 + opposing_weight)),
                    contradicting_evidence_ids=tuple(sorted(set(
                        opponent.contradicting_evidence_ids + tuple(item.id for item in support)
                    ))),
                    status=new_status, updated_at=record.timestamp,
                )
                self._upsert_entry(
                    changed, transaction_id=transaction_id,
                    action="supersede" if explicit_override else "contradict",
                    previous=opponent,
                )
            self._upsert_entry(
                entry, transaction_id=transaction_id,
                action="create" if existing is None else "reinforce", previous=existing,
            )
            action = "create" if existing is None else "reinforce"
            reason = "explicit authority override" if explicit_override else "promotion threshold reached"
            promoted_entry_id = entry.id
        elif existing is not None and opposing:
            lowered = replace(
                existing,
                confidence=max(0.0, existing.confidence - opposing_weight / (1 + opposing_weight)),
                contradicting_evidence_ids=tuple(sorted(set(
                    existing.contradicting_evidence_ids + tuple(item.id for item in opposing)
                ))),
                status=(
                    PersonalityStatus.CONTRADICTED.value
                    if existing.confidence < 0.5 else existing.status
                ),
                updated_at=record.timestamp,
            )
            self._upsert_entry(
                lowered, transaction_id=transaction_id, action="lower_confidence",
                previous=existing,
            )
            action = "lower_confidence"
            reason = "contradictory evidence recorded"
            promoted_entry_id = existing.id
        self._mark_dirty(record.timestamp)
        self._connection.commit()
        self._readonly()
        return PromotionDecision(
            evidence_id=record.id, promoted=promotable,
            entry_id=promoted_entry_id, action=action,
            promotion_score=net_score, threshold=policy.promotion_threshold,
            reason=reason,
        )

    def changes(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._connection.execute(
            "SELECT * FROM changes ORDER BY sequence"
        )]

    def undo_last(self, *, timestamp: str | None = None) -> str:
        last = self._connection.execute(
            "SELECT transaction_id FROM changes ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if last is None:
            raise ValueError("personality package has no reversible changes")
        transaction_id = str(last[0])
        rows = self._connection.execute(
            "SELECT * FROM changes WHERE transaction_id=? ORDER BY sequence DESC",
            (transaction_id,),
        ).fetchall()
        self._writable()
        for row in rows:
            before = row[5]
            if before is None:
                self._connection.execute("DELETE FROM entries WHERE id=?", (row[4],))
            else:
                data = json.loads(before)
                data["supporting_evidence_ids"] = tuple(data["supporting_evidence_ids"])
                data["contradicting_evidence_ids"] = tuple(data["contradicting_evidence_ids"])
                restored = PersonalityEntry(**data)
                self._connection.execute(
                    f"INSERT OR REPLACE INTO entries({ENTRY_COLUMNS}) VALUES ({','.join('?' for _ in range(20))})",
                    _entry_to_row(restored),
                )
        self._connection.execute("DELETE FROM changes WHERE transaction_id=?", (transaction_id,))
        self._mark_dirty(timestamp or _utc_now())
        self._connection.commit()
        self._readonly()
        return transaction_id

    def bulk_insert_entries(
        self, entries: Iterable[PersonalityEntry], *, updated_at: str
    ) -> int:
        """Benchmark/import path; it never bypasses canonical schema validation."""
        rows = []
        for entry in entries:
            PersonalityType(entry.entry_type)
            PersonalityStatus(entry.status)
            EvidenceAuthority(entry.source_authority)
            rows.append(_entry_to_row(entry))
        self._writable()
        self._connection.executemany(
            f"INSERT INTO entries({ENTRY_COLUMNS}) VALUES ({','.join('?' for _ in range(20))})",
            rows,
        )
        self._mark_dirty(updated_at)
        self._connection.commit()
        self._readonly()
        return len(rows)


class PersonalityRouter:
    """Canonical CPU/disk router; no model hidden width or CUDA state."""

    def __init__(
        self,
        *,
        acceptance_threshold: float = 5.0,
        candidate_limit: int = 512,
        entity_width: int = 128,
    ) -> None:
        self.acceptance_threshold = acceptance_threshold
        self.candidate_limit = candidate_limit
        self.encoder = ByteEntityEncoder(entity_width)

    @staticmethod
    def _header_bytes(rows: Sequence[sqlite3.Row]) -> int:
        return sum(len(_canonical_json(tuple(row)).encode()) for row in rows)

    def route(
        self, package: PersonalityPackage, query: PersonalityQuery, *, top_k: int = 4
    ) -> PersonalityRoute:
        if top_k not in (1, 4, 8):
            raise ValueError("P-package router supports top_k 1, 4, or 8")
        rows = package.routing_headers(
            subject=query.subject,
            scopes=(query.interaction_type, query.domain),
            relationship=query.relationship,
            candidate_limit=self.candidate_limit,
        )
        if not rows:
            return PersonalityRoute((), (), 0, 0, 0)
        query_semantic = self.encoder.encode_one(
            " ".join(filter(None, (
                query.subject, query.relation or "", query.interaction_type,
                query.domain, query.relationship or "",
            )))
        )
        now = _parse_time(query.timestamp) if query.timestamp else datetime.now(timezone.utc)
        scored: list[tuple[float, str]] = []
        for row in rows:
            semantic = float(F.cosine_similarity(
                query_semantic,
                self.encoder.encode_one(" ".join((row[2], row[3], row[4]))),
                dim=0,
            ))
            identity = 1.0 if row[2].casefold() == query.subject.casefold() else 0.15
            if row[5] == query.interaction_type:
                context = 1.0
            elif row[5] == query.domain:
                context = 0.85
            elif row[5] == "global":
                context = 0.4
            else:
                context = 0.05
            if row[6] is None:
                relationship = 0.35
            elif query.relationship and row[6].casefold() == query.relationship.casefold():
                relationship = 1.0
            else:
                relationship = 0.0
            relation = 0.5
            if query.relation:
                relation = 1.0 if row[3].casefold() == query.relation.casefold() else 0.0
            age_days = max(0.0, (now - _parse_time(row[10])).total_seconds() / 86400)
            freshness = math.exp(-age_days / 3650)
            score = (
                1.5 * semantic + 2.0 * identity + 2.0 * context
                + 1.5 * relationship + relation + float(row[7])
                + float(row[8]) + float(row[9]) + 0.25 * freshness
            )
            scored.append((score, str(row[0])))
        scored.sort(key=lambda item: (-item[0], item[1]))
        accepted = [item for item in scored if item[0] >= self.acceptance_threshold][:top_k]
        return PersonalityRoute(
            entry_ids=tuple(item[1] for item in accepted),
            scores=tuple(item[0] for item in accepted),
            candidate_count=len(rows), header_bytes_read=self._header_bytes(rows),
            accepted=len(accepted),
        )

    def retrieve(
        self, package: PersonalityPackage, query: PersonalityQuery, *, top_k: int = 4
    ) -> PersonalitySelection:
        import time

        started = time.perf_counter()
        route = self.route(package, query, top_k=top_k)
        entries = tuple(package.entry(entry_id) for entry_id in route.entry_ids)
        entry_bytes = sum(len(_canonical_json(asdict(entry)).encode()) for entry in entries)
        return PersonalitySelection(
            entries=entries, route=route, entry_bytes_read=entry_bytes,
            retrieval_latency_seconds=time.perf_counter() - started,
        )


class FactorizedPersonalityCanonicalizer:
    """Model-independent personality entry -> existing canonical-P protocol."""

    def __init__(
        self,
        representation: FactorizedStateRepresentation,
        *,
        value_labels: Sequence[str],
    ) -> None:
        self.representation = representation.cpu().eval()
        self.value_labels = tuple(value_labels)

    @staticmethod
    def _index(value: str, size: int) -> int:
        return int.from_bytes(hashlib.sha256(value.casefold().encode()).digest()[:8], "big") % size

    def ids(self, entry: PersonalityEntry) -> tuple[int, int, int, int]:
        value_lookup = {label.casefold(): index for index, label in enumerate(self.value_labels)}
        relation_aliases = {
            "owner": 0,
            "preferred_persona": 0,
            "response_style": 0,
            "location": 1,
            "status": 2,
        }
        return (
            self._index(entry.subject, self.representation.entity.num_embeddings),
            relation_aliases.get(
                entry.relation.casefold(),
                self._index(entry.relation, self.representation.relation.num_embeddings),
            ),
            value_lookup.get(
                entry.value.casefold(),
                self._index(entry.value, self.representation.value.num_embeddings),
            ),
            CANONICAL,
        )

    def encode(self, entry: PersonalityEntry) -> tuple[Tensor, tuple[int, int, int, int]]:
        ids = self.ids(entry)
        fields = [torch.tensor([value]) for value in ids]
        with torch.inference_mode():
            vector = self.representation.encode(*fields)[0].detach().cpu()
        return vector, ids


@dataclass(frozen=True)
class PersonalityActivation:
    selection: PersonalitySelection
    store: CanonicalPStore | None
    canonical_bytes: int
    inactive_vram_bytes: int = 0


class PersonalityTranslateSession:
    """Validate once, then reuse one read connection for bounded top-k activation."""

    def __init__(
        self,
        package_path: str | Path,
        router: PersonalityRouter,
        canonicalizer: FactorizedPersonalityCanonicalizer,
        *,
        validate_on_open: bool = True,
    ) -> None:
        self.package_path = Path(package_path)
        self.router = router
        self.canonicalizer = canonicalizer
        self.validated_checksum: str | None = None
        self._validated_stat: tuple[int, int] | None = None
        if validate_on_open:
            with PersonalityPackage(self.package_path, validate=True) as package:
                self.validated_checksum = package.metadata()["content_sha256"]
            stat = self.package_path.stat()
            self._validated_stat = (stat.st_size, stat.st_mtime_ns)
        self._package = PersonalityPackage(self.package_path, validate=False)

    def close(self) -> None:
        self._package.close(checkpoint=False)

    def __enter__(self) -> "PersonalityTranslateSession":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def activate(
        self,
        query: PersonalityQuery,
        *,
        top_k: int = 4,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> PersonalityActivation:
        if self._validated_stat is not None:
            stat = self.package_path.stat()
            if (stat.st_size, stat.st_mtime_ns) != self._validated_stat:
                raise ValueError("personality package changed after integrity validation")
        # Full semantic validation is a cold-open operation. Per-generation
        # activation reuses the read connection and reads bounded router headers
        # plus selected full rows.
        selection = self.router.retrieve(self._package, query, top_k=top_k)
        if not selection.entries:
            return PersonalityActivation(selection, None, 0)
        capacity = len(selection.entries)
        store = CanonicalPStore(CanonicalPConfig(
            slots=capacity, width=512, dtype=dtype, device=device,
            merge_similarity=1.0,
        ))
        for entry in selection.entries:
            vector, ids = self.canonicalizer.encode(entry)
            store.create(
                vector, entity_id=ids[0], relation_id=ids[1], value_id=ids[2],
                metadata_id=ids[3], slot_type=SlotType.FACT,
                confidence=entry.confidence, importance=entry.importance,
                freshness=Freshness.FRESH, persistence=Persistence.DURABLE,
                source=SlotSource.CONVERSATION, label=entry.subject,
            )
        tensors = (
            store.cache.values, store.cache.valid, store.cache.slot_type,
            store.cache.confidence, store.cache.importance, store.cache.freshness,
            store.cache.persistence, store.cache.last_updated, store.cache.source,
            store.entity_id, store.relation_id, store.value_id,
            store.canonical_metadata_id,
        )
        canonical_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return PersonalityActivation(selection, store, canonical_bytes)


def merge_active_personality_with_p_cache(
    p_cache: CanonicalPStore | None, personality: CanonicalPStore | None
) -> CanonicalPStore:
    """Construct a temporary combined view without mutating either sibling."""
    p_count = 0 if p_cache is None else p_cache.cache.occupied
    personality_count = 0 if personality is None else personality.cache.occupied
    capacity = max(1, p_count + personality_count)
    if personality is not None:
        device = personality.cache.device
        dtype = personality.cache.values.dtype
    elif p_cache is not None:
        device = p_cache.cache.device
        dtype = p_cache.cache.values.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float16
    merged = CanonicalPStore(CanonicalPConfig(
        slots=capacity, width=512, dtype=dtype,
        device=device, merge_similarity=1.0,
    ))
    for source in (p_cache, personality):
        if source is None:
            continue
        for index in source.valid.nonzero(as_tuple=False).flatten().tolist():
            merged.create(
                source.canonical_values[index], entity_id=int(source.entity_id[index]),
                relation_id=int(source.relation_id[index]), value_id=int(source.value_id[index]),
                metadata_id=int(source.canonical_metadata_id[index]),
                slot_type=SlotType(int(source.cache.slot_type[index])),
                confidence=float(source.cache.confidence[index]),
                importance=float(source.cache.importance[index]),
                freshness=Freshness(int(source.cache.freshness[index])),
                persistence=Persistence(int(source.cache.persistence[index])),
                source=SlotSource(int(source.cache.source[index])),
                label=source.cache.labels[index],
            )
    return merged


def evidence_from_p_cache(
    store: CanonicalPStore,
    slot: int,
    *,
    evidence_id: str,
    entry_type: str,
    relation: str,
    value: str,
    context: str,
    scope: str,
    timestamp: str,
    archive_reference: str,
    behavioral: bool,
    relationship: str | None = None,
) -> EvidenceRecord | None:
    """Explicitly gated P->evidence flow; ordinary mutable facts return None."""
    if not behavioral or not bool(store.valid[slot]):
        return None
    return EvidenceRecord(
        id=evidence_id, entry_type=entry_type,
        subject=store.cache.labels[slot] or f"entity:{int(store.entity_id[slot])}",
        relation=relation, value=value, context=context, scope=scope,
        confidence=float(store.cache.confidence[slot]),
        source_authority=EvidenceAuthority.SINGLE_OBSERVED.value,
        timestamp=timestamp, archive_reference=archive_reference,
        relationship=relationship,
    )


def synthetic_entry(
    index: int,
    *,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> PersonalityEntry:
    """Deterministic growth-benchmark fixture."""
    evidence_id = f"archive-evidence-{index}"
    return PersonalityEntry(
        id=f"personality-{index:08d}",
        entry_type=PersonalityType.CONTEXTUAL_TENDENCY.value,
        subject=f"subject-{index % 4096}", relation=f"trait-{index % 97}",
        value=f"value-{index}", scope=f"domain-{index % 31}",
        relationship=None, strength=0.75, confidence=0.8,
        importance=0.5, evidence_count=3, context_diversity=2,
        last_reinforced=timestamp, created_at=timestamp, updated_at=timestamp,
        source_authority=EvidenceAuthority.REPEATED_OBSERVED.value,
        supporting_evidence_ids=(evidence_id,), contradicting_evidence_ids=(),
    )
