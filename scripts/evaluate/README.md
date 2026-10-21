# Evaluation entry point

Use [model evaluation](../../docs/workflows/model_evaluation.md) for trial parameters,
stop reasons, result statistics, and runner registration; model-only contracts and exceptions
are in [model specifics](../../docs/workflows/model_specific_evaluation.md).

Run from the repository root after the [environment setup](../../docs/workflows/environment.md):

```bash
pixi run --as-is evaluate-policy -- --help
pixi run --as-is evaluate-policy -- --list-support
```
