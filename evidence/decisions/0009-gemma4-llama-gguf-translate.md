# Gemma4 Q8 llama.cpp compatibility proof

Date: 2026-08-22

Status: bounded causal compatibility proof passed. Phase C was not started.

## Decision

The canonical P protocol and universal router remain unchanged. A new model-specific `.translate` variant maps a routed 512-wide canonical value to a small scalar and a Gemma-native residual direction. llama.cpp consumes the temporary residual through its public control-vector API at Gemma4 layer 41. The quantized base GGUF remains frozen and unchanged.

This adapter does not use LoRA, prompt tokens, self-attention KV, or base-weight training. It contains no P-cache contents or conversation state. Model-native state exists only inside the model-specific `.translate` weights and the temporary runtime control vector.

## Runtime boundary

The supplied llama.cpp build is version 10276 at commit `6ea215d17`. The model reports architecture `gemma4`, 42 layers, hidden width 2560, a 131,072-token configured context, and Q8_0 file type. The tested GGUF SHA-256 is `a4c4177f9fd7e3f56522675afb742f079a53f9226195b7db5e9888c872f053da`.

`llama-server` does not expose intermediate hidden states. The existing universal router therefore receives an explicit canonical entity and relation query before the Gemma adapter runs. The Gemma proof does not claim model-hidden to canonical query projection. This remains a portability limitation.

The stock `llama-cvector-generator` was attempted and rejected for this model because its callback asserted when Gemma4 did not expose the expected `n_layers - 1` hidden tensors. No llama.cpp source or base-model weight was modified. The selected adapter instead uses the frozen Q8 token rows for Alice and Bob to define one normalized model direction. A minimum-norm affine projection maps the corresponding canonical P values to measured runtime strengths `+2` and `-5`.

## Artifact

The selected file is `artifacts/gemma4-e4b-q8-llama.translate`.

- Format: `pcm-llama-gguf-translate-v1`
- Architecture: `canonical_scalar_to_control_vector_v1`
- Parameters: 3,073
- File size: 13,356 bytes
- File SHA-256: `2a25f1641f796d520250d0367dd3b546579a2050c5ef84e2cfff7e598d299785`
- Canonical protocol: `pcm-canonical-p-v1`
- Model hidden width: 2,560
- Attachment layer: 41
- Supported proof values: canonical value IDs 0 and 1, mapped to Gemma tokens 32858 and 15943
- Inactive VRAM: 0 bytes
- Active llama.cpp control buffer: 419,840 bytes
- Relevant model-vector copy: 10,240 bytes
- Extra prompt tokens: 0
- Recent-KV source tokens: 0

The narrow two-value support is intentional for this first quantized runtime proof and must not be described as open-vocabulary semantic portability.

## Matched causal result

The prompt was `The silver key currently belongs to`. The seven prompt tokens were identical in every condition and contained no source-state assertion.

| Condition | Router | Gate | Alice logit | Bob logit | Generated | KL from base | Maximum logit difference from base |
|---|---:|---:|---:|---:|---|---:|---:|
| P disabled | rejected | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |
| Alice state | accepted | 1 | 26.0794 | 0.9100 | ` Alice` | 0.6043 | 11.1606 |
| Bob state | accepted | 1 | 7.2262 | 26.6650 | ` Bob` | 9.6385 | 23.8089 |
| Wrong entity | rejected | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |
| Wrong relation | rejected | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |
| Historical | rejected | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |
| Invalidated | rejected | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |
| Router disabled | rejected | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |
| Translator disabled | accepted | 0 | 23.1686 | 12.0706 | ` the` | 0 | 0 |

The supplied `llama-server` independently generated ` Alice` at strength `+2` and ` Bob` at strength `-5`. The direct `libllama` regression recorded raw full-vocabulary logits and exact identity for every inactive path.

Three P-irrelevant natural-RP runs produced maximum logit difference 0 and KL 0 from the disabled baseline. This proves inert rejection. It does not establish broad RP quality while the value channel is open. The relevant Bob condition has KL 9.6385 and is intentionally recorded as a strong targeted intervention.

## Completion boundary

The quantized GGUF causal-consumption test is complete. The result extends the structural portability evidence to a second real frozen runtime and quantized model. It does not close the stronger second-model semantic portability gate because model-hidden query projection, open-vocabulary values, held-out semantic compositions, and broad relevant-state preservation remain unproven for Gemma4.

Evidence is recorded in `artifacts/gemma4-e4b-q8-causal.json`, `benchmarks/gemma_llama_translate.py`, `benchmarks/llama_cpp_causal_runner.cpp`, and `tests/test_llama_gguf_translate.py`.
