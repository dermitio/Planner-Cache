"""Profile the independent `/state` and `/personality` debug actions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time
import tracemalloc
from typing import Callable

import torch

from pcm.planner.canonical import CanonicalPConfig, CanonicalPStore
from pcm.planner.interactive_session import CanonicalStateManager, PersonalityManager
from pcm.planner.personality import ENTRY_COLUMNS, PersonalityStatus, synthetic_entry
from pcm.planner.representation import FactorizedStateRepresentation


STAMP = "2026-01-01T00:00:00+00:00"


def measure(operation: Callable[[], object]) -> tuple[object, dict[str, object]]:
    gc.collect()
    tracemalloc.start()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    value = operation()
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_python_allocation_bytes": peak_bytes,
    }


def populated_state(
    representation: FactorizedStateRepresentation, count: int,
) -> CanonicalStateManager:
    store = CanonicalPStore(CanonicalPConfig(
        slots=max(1, count), width=512, dtype=torch.float32,
        device="cpu", merge_similarity=1.0,
    ))
    if count:
        store.cache.valid[:count] = True
        store.cache.values[:count] = torch.randn(count, 512)
        store.cache.labels[:count] = [f"entity-{index}" for index in range(count)]
        store.entity_id[:count] = torch.arange(count) % 24
        store.relation_id[:count] = torch.arange(count) % 3
        store.value_id[:count] = torch.arange(count) % 36
    return CanonicalStateManager(representation, store=store)


def profile_state(
    representation: FactorizedStateRepresentation, count: int,
) -> dict[str, object]:
    state = populated_state(representation, count)
    snapshot, action = measure(state.snapshot)
    encoded, serialization = measure(
        lambda: json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
    )
    return {
        "configured_slots": max(1, count),
        "active_entries": count,
        "snapshot": action,
        "json_serialization": serialization,
        "response_bytes": len(encoded),
        "bounded_by_configured_capacity": True,
    }


def profile_personality(
    manager: PersonalityManager, count: int, *, page_limit: int,
) -> dict[str, object]:
    current = manager.package.entry_count()
    if current < count:
        manager.package.bulk_insert_entries(
            (synthetic_entry(index, timestamp=STAMP) for index in range(current, count)),
            updated_at=STAMP,
        )

    unbounded, before_action = measure(manager.visible_entries)
    before_json, before_serialization = measure(
        lambda: json.dumps(unbounded, ensure_ascii=False).encode("utf-8")
    )
    page, after_action = measure(
        lambda: manager.visible_entry_page(limit=page_limit)
    )
    after_json, after_serialization = measure(
        lambda: json.dumps(page, ensure_ascii=False).encode("utf-8")
    )
    return {
        "active_entries": count,
        "before_unbounded": {
            "action": before_action,
            "json_serialization": before_serialization,
            "hydrated_entries": len(unbounded),
            "response_bytes": len(before_json),
        },
        "after_bounded": {
            "action": after_action,
            "json_serialization": after_serialization,
            "hydrated_entries": page["returned"],
            "response_bytes": len(after_json),
            "total_active": page["total_active"],
            "truncated": page["truncated"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-limit", type=int, default=100)
    parser.add_argument("--state-counts", default="0,64,128,256,512,1024")
    parser.add_argument("--personality-counts", default="100,1000,10000,100000")
    args = parser.parse_args()

    torch.manual_seed(211)
    representation = FactorizedStateRepresentation(24, 3, 36)
    result: dict[str, object] = {
        "experiment": "planner-cache-debug-actions-profile-v1",
        "page_limit": args.page_limit,
        "state": {},
        "personality": {},
    }
    for count in (int(item) for item in args.state_counts.split(",")):
        result["state"][str(count)] = profile_state(representation, count)
    with tempfile.TemporaryDirectory(prefix="planner-cache-debug-profile-") as directory:
        manager = PersonalityManager(Path(directory) / "profile.ppkg", representation)
        try:
            for count in (int(item) for item in args.personality_counts.split(",")):
                result["personality"][str(count)] = profile_personality(
                    manager, count, page_limit=args.page_limit,
                )
            result["personality_query_plans"] = {
                "active_count": [
                    str(row[3]) for row in manager.package._connection.execute(
                        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM entries WHERE status=?",
                        (PersonalityStatus.ACTIVE.value,),
                    )
                ],
                "bounded_page": [
                    str(row[3]) for row in manager.package._connection.execute(
                        f"EXPLAIN QUERY PLAN SELECT {ENTRY_COLUMNS} FROM entries "
                        "WHERE status=? ORDER BY id LIMIT ? OFFSET ?",
                        (PersonalityStatus.ACTIVE.value, args.page_limit, 0),
                    )
                ],
            }
        finally:
            manager.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
