# Tensor Translation Layer

TTL means Tensor Translation Layer. Its extension is `.ttl` and its support level is semantic or internal.

```text
canonical P -> universal router -> .ttl -> model hidden state -> frozen model
```

The reference Pythia TTL maps model hidden state to a factorized canonical query. It translates a selected 512-wide canonical value back to the model's 2,048-wide hidden space and applies a relevance-controlled residual at layer 23. The unchanged package has 2,707,464 learned parameters. The Pythia base is frozen and receives zero gradients.

`planner-cache-ttl-v1` records the adapter class, support level, canonical protocol, model identity, hidden width, attachment layers, architecture configuration, and a tensor checksum. It contains no base weights, P-cache contents, conversation state, prompt tokens, attention KV, or optimizer state.

The migrated artifact SHA-256 is `72ef68d07ee27c37b90432d34d4be5c2c280ae1bcb08236e37a0e458c054d8d7`.

```python
from pcm.planner import TensorTranslationLayer

ttl = TensorTranslationLayer.load(
    "artifacts/pythia-1.4b-final-layer.ttl",
    device="cuda",
)
ttl.validate_compatibility(
    model_id="pythia-1.4b",
    model_hidden_width=2048,
    attachment_layers=(23,),
)
```

Supporting another decoder requires a model-specific attachment implementation and a trained TTL. GPT-NeoX layer names are Pythia adapter details and are not part of the TTL file contract.

Legacy Pythia `.translate` files can be loaded through `TensorTranslationLayer.load`. They emit a deprecation warning and are explicitly classified as TTL. Other historical `.translate` variants are research-only and are not silently promoted.
