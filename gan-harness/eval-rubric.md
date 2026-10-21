# Code-only evaluation rubric

- Raw units and every conversion are explicit.
- No synthetic label silently replaces missing D2Los metadata.
- Data splits prevent map leakage and are persisted.
- The synthesizer is batched, differentiable and returns complex64 CSI.
- Conditions are recomputed from generated paths.
- K-factor validity is represented explicitly.
- Training, generation and evaluation are separate entry points.
- Checkpoints contain full resume state and configuration.
- Tests exist but are not executed during the local code-generation task.
