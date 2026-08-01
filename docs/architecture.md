# Architecture

The project keeps the research workflow explicit:

```text
configuration
    -> dataset preparation
    -> model adapters
    -> cached embeddings
    -> category-aware retrieval
    -> expert-rule retrieval
    -> evidence-aware reranking
    -> constrained explanations
    -> recommendation and faithfulness evaluation
```

Each stage receives typed configuration and returns ordinary Python or pandas objects.
Provider-specific code is isolated in `models/`, while methodological code remains
provider-independent.

## Design principles

1. Experimental choices live in YAML; implementation mechanics live in Python.
2. Every expensive artifact is keyed by its material inputs.
3. Every run stores its resolved configuration and environment manifest.
4. Model adapters make model replacement explicit rather than relying on hidden globals.
5. Evaluation cases, negative samples, generations, and judge responses are reusable
   artifacts so comparisons operate on identical inputs.
6. The final test protocol should be frozen before the final paper/thesis run.

## Package map

- `config.py`: strict schemas, inheritance, and command-line overrides.
- `cache.py`: content-addressed artifacts and provenance sidecars.
- `data/`: dataset adapters and category preparation.
- `models/`: text, multimodal, captioning, generator, and judge providers.
- `retrieval.py`: category-aware indexes and nearest-neighbour retrieval.
- `evidence.py`: knowledge-base validation and rule retrieval.
- `reranking.py`: evidence-aware score composition.
- `generation.py`: candidate-locked grounding variants.
- `evaluation/`: ranking, explanation, statistical, and human-review protocols.
- `run.py`: deterministic setup and immutable run directories.
- `cli.py`: stable user-facing commands.

