"""Profile checksum, SQLite, routing, hydration, and canonicalization costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcm.planner.personality_profile import profile_ppkg_lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--package", type=Path, default=Path("artifacts/ppkg-profile-100k.ppkg")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/ppkg-100k-profile.json")
    )
    args = parser.parse_args()
    result = profile_ppkg_lookup(
        args.package, entries=args.entries, repeats=args.repeats
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "normal_query_checksum_calls": result["normal_query_checksum_calls"],
        "normal_push_checksum_calls": result["evidence_push"]["normal_push_checksum_calls"],
        "modeled_speedup": result["modeled_speedup"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
