"""Generate publication assets from recorded Planner Cache artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cairosvg


ASSETS = Path(__file__).resolve().parent
PACK_ARTIFACTS = ASSETS.parent / "artifacts"
ROOT = ASSETS.parent if PACK_ARTIFACTS.is_dir() else ASSETS.parents[1]
ARTIFACTS = ROOT / "artifacts"


def load(name: str):
    return json.loads((ARTIFACTS / name).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name: str, fields: list[str], rows: list[dict[str, object]]) -> None:
    with (ASSETS / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def architecture_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#385170"/></marker>
</defs>
<rect width="1400" height="820" fill="#f8fafc"/>
<text x="700" y="54" text-anchor="middle" font-family="Arial,sans-serif" font-size="34" font-weight="700" fill="#12233f">Planner Cache publication architecture</text>
<text x="700" y="86" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" fill="#50627a">Portable bounded semantic state for frozen language models</text>
<rect x="45" y="125" width="1310" height="220" rx="22" fill="#edf4fb" stroke="#7da2c8" stroke-width="2"/>
<text x="75" y="160" font-family="Arial,sans-serif" font-size="20" font-weight="700" fill="#244d76">Model and runtime owned</text>
<rect x="90" y="195" width="260" height="105" rx="16" fill="#ffffff" stroke="#2c7a7b" stroke-width="3"/>
<text x="220" y="235" text-anchor="middle" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#1f5960">Recent KV</text>
<text x="220" y="266" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#50627a">Exact recent wording</text>
<rect x="570" y="195" width="260" height="105" rx="16" fill="#ffffff" stroke="#315c9b" stroke-width="3"/>
<text x="700" y="235" text-anchor="middle" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#244d76">Frozen model</text>
<text x="700" y="266" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#50627a">Pythia and Gemma proof paths</text>
<rect x="1050" y="195" width="260" height="105" rx="16" fill="#ffffff" stroke="#6856a5" stroke-width="3"/>
<text x="1180" y="229" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="700" fill="#55428e">History and tools</text>
<text x="1180" y="259" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Archive and retrieval systems</text>
<text x="1180" y="282" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Outside Planner Cache core</text>
<rect x="45" y="390" width="1310" height="350" rx="22" fill="#fff7ed" stroke="#cf8a49" stroke-width="2"/>
<text x="75" y="425" font-family="Arial,sans-serif" font-size="20" font-weight="700" fill="#91501e">Planner Cache owned</text>
<rect x="85" y="442" width="245" height="32" rx="12" fill="#fffdf8" stroke="#b36a2e" stroke-width="2" stroke-dasharray="7 5"/>
<text x="207" y="464" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" font-weight="700" fill="#8f4d1e">Hidden post-turn review</text>
<rect x="85" y="485" width="245" height="125" rx="16" fill="#ffffff" stroke="#b36a2e" stroke-width="3"/>
<text x="207" y="527" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#8f4d1e">P-cache</text>
<text x="207" y="558" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Bounded mutable state</text>
<text x="207" y="582" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Canonical P protocol</text>
<rect x="405" y="485" width="245" height="125" rx="16" fill="#ffffff" stroke="#3e7c59" stroke-width="3"/>
<text x="527" y="527" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#2d6848">Universal router</text>
<text x="527" y="558" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Canonical selection</text>
<text x="527" y="582" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Model independent</text>
<rect x="725" y="485" width="245" height="125" rx="16" fill="#ffffff" stroke="#315c9b" stroke-width="3"/>
<text x="847" y="527" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#244d76">.ttl / .ltl</text>
<text x="847" y="558" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Compatibility boundary</text>
<text x="847" y="582" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Semantic / lexical</text>
<rect x="1045" y="485" width="245" height="125" rx="16" fill="#ffffff" stroke="#6856a5" stroke-width="3"/>
<text x="1167" y="527" text-anchor="middle" font-family="Arial,sans-serif" font-size="25" font-weight="700" fill="#55428e">P-package</text>
<text x="1167" y="558" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Disk resident personality</text>
<text x="1167" y="582" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#50627a">Portable .ppkg</text>
<path d="M 330 548 L 395 548" stroke="#385170" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<path d="M 207 474 L 207 482" stroke="#b36a2e" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
<path d="M 700 300 C 650 375 350 385 225 439" stroke="#b36a2e" stroke-width="2" stroke-dasharray="8 6" fill="none" marker-end="url(#arrow)"/>
<path d="M 650 548 L 715 548" stroke="#385170" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<path d="M 1045 620 C 930 695 585 695 527 620" stroke="#6856a5" stroke-width="3" stroke-dasharray="10 7" fill="none" marker-end="url(#arrow)"/>
<path d="M 847 485 C 820 405 755 335 710 302" stroke="#315c9b" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
<path d="M 350 247 L 558 247" stroke="#2c7a7b" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
<text x="700" y="782" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#50627a">Canonical files never store base weights, token IDs, conversation text, or model-native hidden vectors</text>
</svg>
"""


def line_plot_svg(title: str, subtitle: str, series: list[tuple[str, str, list[tuple[float, float]]]], x_label: str, y_label: str) -> str:
    width = 1100
    height = 650
    left = 105
    right = 55
    top = 115
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_x = [x for _, _, points in series for x, _ in points]
    all_y = [y for _, _, points in series for _, y in points]
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(0.0, min(all_y))
    y_max = max(all_y) * 1.08

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    parts.append('<rect width="1100" height="650" fill="#ffffff"/>')
    parts.append(f'<text x="550" y="42" text-anchor="middle" font-family="Arial,sans-serif" font-size="28" font-weight="700" fill="#12233f">{title}</text>')
    parts.append(f'<text x="550" y="72" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#50627a">{subtitle}</text>')
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dbe4ee" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="#50627a">{value:.1f}</text>')
    x_ticks = sorted(set(all_x))
    for value in x_ticks:
        x = px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_height}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_height+27}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#50627a">{int(value)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#385170" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#385170" stroke-width="2"/>')
    for index, (label, color, points) in enumerate(series):
        coordinates = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y in points:
            parts.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="5" fill="{color}"/>')
        legend_x = left + index * 280
        parts.append(f'<line x1="{legend_x}" y1="{height-30}" x2="{legend_x+32}" y2="{height-30}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{legend_x+42}" y="{height-25}" font-family="Arial,sans-serif" font-size="14" fill="#28384e">{label}</text>')
    parts.append(f'<text x="{left+plot_width/2}" y="{height-55}" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#28384e">{x_label}</text>')
    parts.append(f'<text x="28" y="{top+plot_height/2}" text-anchor="middle" transform="rotate(-90 28 {top+plot_height/2})" font-family="Arial,sans-serif" font-size="15" fill="#28384e">{y_label}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    audit = load("active-system-audit.json")
    cuda = load("active-system-cuda-attribution.json")
    split = load("phase-b-split-translator.json")
    personality = load("phase-b-personality-package.json")
    gemma = load("gemma4-e4b-q8-causal.json")
    vram = load("vram-comparison.json")

    router_rows = []
    for count, values in sorted(audit["router_scaling"]["measurements"].items(), key=lambda item: int(item[0])):
        router_rows.append({
            "slots": int(count),
            "top1_accuracy": values["top1_accuracy"],
            "top4_recall": values["top4_recall"],
            "mrr": values["mrr"],
            "routing_latency_ms": values["latency_seconds"] * 1000,
        })
    write_csv("router_scaling.csv", list(router_rows[0]), router_rows)

    ppkg_rows = []
    for count, values in sorted(audit["ppkg_scaling"].items(), key=lambda item: int(item[0])):
        ppkg_rows.append({
            "entries": int(count),
            "disk_bytes": values["disk_bytes"],
            "checksum_ms": values["checksum_seconds"] * 1000,
            "db_open_ms": values["db_open_seconds"] * 1000,
            "routing_header_ms": values["routing_header_wall_seconds"] * 1000,
            "row_hydration_ms": values["row_hydration_wall_seconds"] * 1000,
            "canonical_conversion_ms": values["canonical_conversion_wall_seconds"] * 1000,
            "candidate_headers": values["candidate_headers"],
            "entries_loaded": values["entries_loaded"],
            "logical_bytes_read": values["logical_bytes_read"],
            "inactive_vram_bytes": values["inactive_vram_bytes"],
        })
    write_csv("ppkg_scaling.csv", list(ppkg_rows[0]), ppkg_rows)

    causal_rows = []
    for condition, values in cuda["causal"].items():
        causal_rows.append({
            "condition": condition,
            "selected_state": values["selected_state"] or "",
            "router_score": "" if values["router_score"] is None else values["router_score"],
            "router_accepted": values["router_accepted"],
            "gate": values["gate"],
            "alice_logit": values["alice_logit"],
            "bob_logit": values["bob_logit"],
            "generated": values["generated"].replace("\n", "\\n"),
            "kl_from_base": values["kl_from_base"],
            "latency_ms": cuda["latency_seconds"][condition] * 1000,
        })
    write_csv("causal_conditions.csv", list(causal_rows[0]), causal_rows)

    gemma_rows = []
    for condition, values in gemma["conditions"].items():
        gemma_rows.append({
            "condition": condition,
            "router_accepted": values["router_accepted"],
            "gate": values["gate"],
            "strength": values["strength"],
            "alice_logit": values["alice_logit"],
            "bob_logit": values["bob_logit"],
            "alice_probability": values["alice_probability"],
            "bob_probability": values["bob_probability"],
            "generated": values["generated"].replace("\n", "\\n"),
            "kl_from_base": values["kl_from_base"],
            "max_abs_logit_difference_from_base": values["max_abs_logit_difference_from_base"],
            "latency_ms": values["latency_ms"],
        })
    write_csv("gemma_causal_conditions.csv", list(gemma_rows[0]), gemma_rows)

    vram_rows = []
    for values in vram["results"]:
        vram_rows.append({
            "workload_tokens": values["prompt_tokens"],
            "condition": values["condition"],
            "generated_tokens": values["generated_tokens"],
            "p_cache_slots": values["p_cache_slots"],
            "p_cache_canonical_bytes": values["p_cache_canonical_bytes"],
            "retained_kv_cache_bytes": values["retained_kv_cache_bytes"],
            "baseline_allocated_bytes": values["baseline_allocated_bytes"],
            "peak_allocated_bytes": values["peak_allocated_bytes"],
            "peak_reserved_bytes": values["peak_reserved_bytes"],
            "incremental_peak_allocated_bytes": values["incremental_peak_allocated_bytes"],
            "incremental_peak_reserved_bytes": values["incremental_peak_reserved_bytes"],
            "runtime_seconds": values["runtime_seconds"],
            "failure": "" if values["failure"] is None else values["failure"]["type"],
        })
    write_csv("vram_comparison.csv", list(vram_rows[0]), vram_rows)

    architecture = architecture_svg()
    (ASSETS / "architecture.svg").write_text(architecture)
    qpdf = shutil.which("qpdf")
    if qpdf is None:
        raise RuntimeError("qpdf is required to normalize publication PDF metadata")
    with tempfile.TemporaryDirectory(prefix="planner-cache-publishing-") as temporary:
        raw_pdf = Path(temporary) / "architecture.raw.pdf"
        cairosvg.svg2pdf(bytestring=architecture.encode(), write_to=str(raw_pdf))
        subprocess.run(
            [qpdf, "--remove-info", "--remove-metadata", "--deterministic-id", str(raw_pdf), str(ASSETS / "architecture.pdf")],
            check=True,
        )
    (ASSETS / "architecture.mmd").write_text("""flowchart LR
subgraph Runtime[Model and runtime owned]
KV[Recent KV]
LM[Frozen model]
EXT[History and tool systems]
end
subgraph Planner[Planner Cache owned]
P[P-cache]
R[Universal router]
C{Compatibility boundary}
N[Native P]
TTL[.ttl semantic/internal]
LTL[.ltl lexical/output]
PKG[P-package .ppkg]
REVIEW[Hidden post-turn review]
end
KV --> LM
LM -. side-channel after visible reply .-> REVIEW --> P
P --> R --> C
C --> N --> LM
C --> TTL --> LM
C --> LTL --> LM
PKG -. selected canonical state .-> R
EXT -. external evidence .-> LM
""")

    router_points = [(float(row["slots"]), float(row["top1_accuracy"]) * 100) for row in router_rows]
    legacy = audit["router_scaling"]["immutable_pre_fix_baseline"]
    legacy_points = [(float(key), float(value) * 100) for key, value in sorted(legacy.items(), key=lambda item: int(item[0]))]
    (ASSETS / "router_scaling.svg").write_text(line_plot_svg(
        "Canonical router scaling",
        "Post-audit identity-safe storage compared with the immutable pre-fix baseline",
        [("Post-audit top-1", "#2c7a7b", router_points), ("Pre-fix top-1", "#b36a2e", legacy_points)],
        "Configured P slots",
        "Top-1 accuracy percent",
    ))

    ppkg_points = [(float(row["entries"]), float(row["routing_header_ms"])) for row in ppkg_rows]
    (ASSETS / "ppkg_lookup.svg").write_text(line_plot_svg(
        "P-package indexed lookup",
        "Bounded header routing while package contents grow on disk",
        [("Routing header latency", "#6856a5", ppkg_points)],
        "Package entries",
        "Latency ms",
    ))

    condition_labels = (
        ("p_cache_only", "P-cache only", "#b36a2e"),
        ("kv_only", "Retained KV only", "#2c7a7b"),
        ("p_cache_plus_kv", "P-cache plus KV", "#6856a5"),
    )
    vram_series = []
    for condition, label, color in condition_labels:
        points = [
            (
                float(row["workload_tokens"]),
                float(row["incremental_peak_allocated_bytes"]) / (1024 * 1024),
            )
            for row in vram_rows if row["condition"] == condition
        ]
        vram_series.append((label, color, points))
    (ASSETS / "vram_comparison.svg").write_text(line_plot_svg(
        "Matched P-cache and retained-KV VRAM",
        "Frozen Pythia-1.4B, batch 1, float16 base, 8 greedy generated tokens",
        vram_series,
        "Prompt tokens and configured P slots",
        "Incremental peak allocated MiB",
    ))

    manifest = {
        "generated_from": {
            name: sha256(ARTIFACTS / name)
            for name in (
                "active-system-audit.json",
                "active-system-cuda-attribution.json",
                "phase-b-factorized-representation.json",
                "phase-b-personality-package.json",
                "phase-b-split-translator.json",
                "ppkg-100k-profile.json",
                "gemma4-e4b-q8-causal.json",
                "gemma-native-prompt-equivalence.json",
                "post-turn-memory-review-acceptance.json",
                "debug-actions-profile.json",
                "vram-comparison.json",
            )
        },
        "public_binary_artifacts": {
            name: sha256(ARTIFACTS / name)
            for name in (
                "canonical-p-v1.router",
                "pythia-1.4b-final-layer.ttl",
                "personality-proof.ppkg",
                "gemma4-e4b-q8-llama.ltl",
            )
        },
        "selected_configuration": {
            "base_model": cuda["base_model"],
            "ttl_parameters": (
                audit["ttl_profile"]["cpu"]["parameter_count"]
                if "ttl_profile" in audit
                else audit["translate_profile"]["cpu"]["parameter_count"]
            ),
            "canonical_protocol": "pcm-canonical-p-v1",
            "router_format": "pcm-canonical-router-v1",
            "ttl_format": "planner-cache-ttl-v1",
            "ppkg_format": personality["format"],
            "ppkg_protocol": personality["protocol"],
            "selected_attachment": split["selected_variant"],
            "gemma_model": gemma["model"]["name"],
            "gemma_ltl_format": "planner-cache-ltl-v1",
            "gemma_ltl_parameters": 0,
        },
    }
    (ASSETS / "EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
