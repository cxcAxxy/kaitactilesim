# Policy deployment entry point

Use [model deployment](../../docs/workflows/model_deployment.md) for prepare/verify/serve/run,
manifest integrity, runtime profiles, metadata readiness, and process cleanup.
Model-only contracts are in [model specifics](../../docs/workflows/model_specific_evaluation.md).

Run from the repository root after the [environment setup](../../docs/workflows/environment.md):

```bash
pixi run --as-is deploy-policy -- --help
pixi run --as-is deploy-policy -- --list-support
```
