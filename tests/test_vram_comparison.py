import json
from pathlib import Path


def test_publication_vram_comparison_is_complete_and_matched():
    artifact = Path("artifacts/vram-comparison.json")
    result = json.loads(artifact.read_text())
    assert result["experiment"] == "planner-cache-matched-vram-comparison-v1"
    assert result["completion"] == {
        "conditions": 9,
        "estimated_values": 0,
        "failed": 0,
        "oom_events": 0,
        "successful": 9,
    }
    shared = result["shared_configuration"]
    assert shared["workload_prompt_and_slot_sizes"] == [64, 256, 1024]
    assert shared["batch_size"] == 1
    rows = {
        (row["prompt_tokens"], row["condition"]): row
        for row in result["results"]
    }
    assert len(rows) == 9
    baselines = {row["baseline_allocated_bytes"] for row in rows.values()}
    assert len(baselines) == 1
    for size in (64, 256, 1024):
        p_only = rows[size, "p_cache_only"]
        kv_only = rows[size, "kv_only"]
        combined = rows[size, "p_cache_plus_kv"]
        assert p_only["retained_kv_cache_bytes"] == 0
        assert p_only["p_cache_canonical_bytes"] > 0
        assert kv_only["retained_kv_cache_bytes"] > 0
        assert kv_only["p_cache_canonical_bytes"] == 0
        assert combined["retained_kv_cache_bytes"] == kv_only["retained_kv_cache_bytes"]
        assert combined["p_cache_canonical_bytes"] == p_only["p_cache_canonical_bytes"]
        assert all(
            row["failure"] is None and row["fallback"] is None
            for row in (p_only, kv_only, combined)
        )
