# Cache behaviour

`cache.policy` accepts:

- `reuse`: load a matching artifact or calculate it if absent.
- `refresh`: recalculate and replace the artifact for the current fingerprint.
- `disabled`: calculate without retaining a shared cache artifact.

Changing a model, revision, dataset source, preprocessing choice, prompt version, or
methodological parameter changes the relevant fingerprint. A run manifest records the
resolved configuration, and artifact metadata records the fingerprint inputs.

Never identify an embedding cache using only a human-readable filename. Two models can
share output dimensions while representing items incompatibly.

