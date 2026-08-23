# General lexical `.translate` trainer experiment

Date: 2026-08-22

Status: trainer and runtime interface implemented. The first Gemma4 Q8 open-vocabulary experiment was rejected. The bounded two-value adapter from Update 0009 remains the selected Gemma compatibility artifact. Phase C was not started.

## Decision

A reusable staged translator trainer now treats the target model as a frozen lexical teacher. It does not change P-cache, the universal router, P-package, the GGUF, or any base-model parameter.

The portable input is the complete UTF-8 value string. Model token identifiers and quantized embedding rows exist only behind the runtime-specific `ModelLexicalTeacher` boundary. They are absent from canonical P state and the deployable adapter.

The reference compact translator is:

```text
UTF-8 bytes
    -> 257-entry byte embedding
    -> width-64 convolutions at kernels 1, 2, and 3
    -> masked mean and maximum pooling
    -> width-256 lexical state
    -> rank-128 projection
    -> 2,560-wide normalized Gemma control direction
```

The deployable artifact contains 504,706 parameters and occupies 2,021,856 bytes. It contains no base weights, model vocabulary table, P contents, conversation state, or optimizer state. The format records the model and runtime identity, model checksum, canonical protocol, lexical encoder version, training vocabulary hash, training configuration hash, attachment layer, hidden width, architecture, parameter count through inspection, and tensor checksum.

`ModelLexicalTeacher` defines vocabulary enumeration, token decoding, frozen target extraction, complete-string composition, and compatibility metadata. Implementations are included for stock GGUF input embeddings and frozen PyTorch embedding layers. The Gemma teacher uses `token_embd.weight` because stock llama.cpp exposes that vocabulary-wide model-native representation but does not expose intermediate hidden states through `llama-server`.

## Training and held-out split

The exact Gemma4 Q8 vocabulary contained 262,144 entries. The teacher marked 255,891 entries usable before duplicate decoded aliases were collapsed. The deterministic complete-string split contained 230,048 training values and 25,514 held-out values. Duplicate tokenizer aliases use one deterministic target because canonical text cannot represent two different token identifiers without violating the tokenizer-independent boundary.

Training used three ordered stages:

1. three epochs over all 230,048 training lexical values
2. three epochs over 8,192 disjoint multi-token compositions
3. two epochs over 2,048 state values with same-prefix, same-suffix, and similar-spelling hard negatives

The fixed open-string evaluation manifest contains 1,120 complete strings. It has 160 values in each of seven categories covering unseen names, numbers, locations, statuses, multi-word values, synthetic strings, and UTF-8 strings. No complete evaluation string appears in translator training.

The universal router remains responsible for wrong entity, wrong relation, historical, invalidated, and irrelevant rejection. The value translator does not absorb those semantics.

## Alignment result

The adapter did not pass lexical generalization.

| Category | Values | Mean cosine | Retrieval accuracy |
|---|---:|---:|---:|
| Held-out vocabulary probe | 4,096 | 0.3351 | 45.56% |
| Locations | 160 | 0.6790 | 13.13% |
| Multi-word | 160 | 0.6752 | 6.25% |
| Numbers | 160 | 0.6592 | 0.63% |
| Statuses | 160 | 0.6911 | 6.25% |
| Synthetic strings | 160 | 0.7338 | 0.63% |
| Unseen names | 160 | 0.6460 | 0.63% |
| UTF-8 strings | 160 | 0.7134 | 2.50% |

The relatively high cosine with poor discrimination shows collapse toward broad embedding neighborhoods. It does not demonstrate recoverable value identity.

## Causal result

The direct stock-libllama runner loaded the frozen GGUF once, applied query-specific translator directions at layer 41, retained identical prompt tokens, and measured full-vocabulary logits. The controlled prompt contained no source-state assertion and introduced no P prompt tokens.

A balanced 35-value diagnostic at strength 1 produced 0% first-token and 0% exact token-sequence accuracy in every category. Mean correct-token logit lift ranged from negative 0.084 for numbers to positive 2.529 for UTF-8 values. Mean KL ranged from 0.0135 to 0.0569. Stronger diagnostic scales increased many target logits but still produced 0% generation and rapidly increased divergence. A seven-value familiar probe containing Alice, Bob, desk, workshop, Nathra, 7319, and İzmir also produced 0% at strength 4.

This localizes the failure at two boundaries:

1. a compact byte mapper does not recover held-out frozen token rows with enough discrimination
2. approximate input-embedding alignment is not a reliable layer-41 causal residual target

The full 1,120-value lexical evaluation was completed. A 1,000-value causal run was not justified after balanced causal diagnostics reached 0% across every category and familiar values. The experiment is therefore rejected rather than described as general-purpose.

## Active status

The immutable `gemma4-e4b-q8-llama.translate` artifact remains active for its measured Alice and Bob proof. `run-gemma.sh` continues to use that bounded adapter by default.

The lexical trainer, format loader, query-specific llama.cpp control-vector integration, deterministic manifests, and causal runner remain research infrastructure. They do not expand the published Gemma support claim.

Evidence is recorded in:

- `train_translate.py`
- `src/pcm/planner/lexical_translate.py`
- `src/pcm/planner/translate_trainer.py`
- `benchmarks/evaluate_lexical_translate.py`
- `benchmarks/llama_cpp_lexical_runner.cpp`
- `artifacts/gemma4-e4b-q8-lexical.translate`
- `artifacts/gemma4-e4b-q8-lexical.json`
- `artifacts/gemma4-e4b-q8-lexical.split.json`
- `artifacts/gemma4-e4b-q8-lexical-causal.json`
- `artifacts/gemma4-e4b-q8-lexical-causal-strength4.json`
- `artifacts/gemma4-e4b-q8-lexical-causal-familiar.json`

No base model or active canonical component was trained or modified.
