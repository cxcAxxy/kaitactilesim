# Unified policy deployment

`deploy.py` is a thin, model-free coordinator around the audited deployment
scripts in `scripts/workcell`. It does not replace task/model validators or
itself import model packages. Sidecars and local workers stay in child
processes; the explicitly advertised `embedded` mode is the exception and loads
its model inside the evaluation runner.

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- prepare --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- verify --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- serve --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- run --help
```

The support listing is authoritative. Each task/model row separately declares
`prepare`, `verify`, `serve`, `run`, and `evaluate`. In particular:

- `websocket-sidecar`: pi0.5, supported EgoSteer deployments, and LingBot use a
  policy server. `run --service-mode managed` owns and cleans up that server;
  `--service-mode external --endpoint ...` only connects and never terminates it.
- `local-worker`: EgoTouch, poker `pi05+cpm`, and supported `ftp1` task runners create a private worker in the declared model
  Python. `serve` is intentionally not applicable.
- `embedded`: Poker pi0.5+T-Rex loads its base and tactile expert in the runner.
  It has no standalone server and does not claim a complete prepare contract.

Poker CPM and FTP-1 freeze native H30/H32, execution K, the model Python and
package versions, checkpoint/normalizer/tokenizer files, source and scene
assets. Their initial evaluations remain diagnostic, separate from historical
H50 reports. See [the tactile workflow](../../docs/workflows/policy_evaluation_video.md#cpm--ftp-1-触觉闭环)
and `configs/runtimes.tactile.toml` for the local environments and commands.
FTP-1 also has diagnostic interfaces for ten shared tasks, with task/domain,
training corpus, tactile target and action-recipe binding. Those tasks currently
await matching trained checkpoints; normalization alone is insufficient.
Non-poker CPM and FTP-1 PickPlace/USB insert remain unregistered.

## Runtime profiles

Model environments can be kept out of commands with a TOML or JSON file:

```toml
default_runtime = "openpi"

[runtimes.openpi]
python = "/path/to/openpi/bin/python"
project_root = "/path/to/openpi"

[runtimes.egotouch]
python = "/path/to/egotouch/bin/python"
project_root = "/path/to/egotouch"

[runtimes.egotouch.env]
CUDA_VISIBLE_DEVICES = "0"
```

Paths are resolved relative to the config. Use
`--runtime-config runtimes.toml --runtime egotouch`. Prepare/run plans and their
pipeline records include the selected config hash and a redacted environment.
Sidecar and local-worker model imports remain in child processes.

## Prepare and verify

Common checkpoint/output arguments are stable. Existing backend-only arguments
go after `--` and are passed to the original prepare script:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- prepare \
  --task whiteboard-wipe --model-family pi05 \
  --checkpoint /path/to/checkpoints/10000 \
  --output-dir /path/to/deployments/whiteboard_10000 \
  --runtime-config runtimes.toml --runtime openpi

/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- verify \
  --deployment-manifest /path/to/deployments/whiteboard_10000/deployment_manifest.json \
  --runtime-config runtimes.toml --runtime openpi
```

Prepare keeps the task/model `deployment_manifest.json` untouched and adds
`pipeline_deployment.json` beside it. The outer record contains the prepare
command, runtime-config identity, original manifest hash/deployment ID, and
verification result. Existing pi0.5 source hashes and training provenance are
checked by `verify_pi05_deployment`; they are not replaced by weaker generic
checks.

## Serve and run

Serve one sidecar in the foreground:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- serve \
  --deployment-manifest /path/to/deployment_manifest.json \
  --runtime-config runtimes.toml --runtime openpi --port 18783
```

Or let `run` supervise it:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- run \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/run_001 \
  --runtime-config runtimes.toml --runtime openpi \
  --service-mode managed --num-trials 1 --video-count 1
```

An external server is never killed by the coordinator:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- run \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/run_002 \
  --service-mode external --endpoint ws://127.0.0.1:18783
```

Readiness is a WebSocket metadata handshake that checks deployment ID,
checkpoint hash, observation contract, action representation, and action
horizon/dimension. It also checks task and model family when the server
advertises them. A listening TCP port alone is not considered ready. Managed
connect endpoints must be loopback `ws://` addresses because the existing
sidecars are local plaintext servers; use external mode for remote or TLS
endpoints. Policy traffic bypasses HTTP proxies without clearing proxy settings
needed for model downloads.

The simulator evaluator keeps the simulator environment even when the sidecar
uses a different model runtime. EgoTouch's local worker instead receives the
selected runtime environment. Owned prepare, validation, server, and evaluation
children run in private process groups and are reclaimed on SIGINT/SIGTERM;
external services are never signalled.

Each run uses a new root containing:

```text
run_root/
  run_manifest.json  # frozen deployment ID/path plus runtime parameters
  server.log         # managed sidecar only
  evaluation/        # existing evaluate-policy output
```

`--execute-steps` is omitted by default so the existing evaluator retains its
model-specific default (notably PickPlace EgoTouch executes 5 from H50).
`--smoke-seconds N` is diagnostic-only: it exercises model I/O and simulation
for a short bound, records `formal_metrics_valid=false`, and does not turn an
unfinished task into a formal success. Extra evaluator flags may follow `--`.
