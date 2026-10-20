# Conversion documentation

The canonical conversion guide is [docs/workflows/data_conversion.md](../../docs/workflows/data_conversion.md).

LeRobot v3 fields and timing are specified in the
[format contract](../../docs/workflows/lerobot_v3.md).
Conversion behavior and compatibility rules are maintained in the canonical guide,
not duplicated here.

Run from the repository root:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run convert-dataset -- --help
```

`--list-support` is the machine-readable adapter catalog. In addition to task,
format and camera support, every row declares its runtime kind, accepted named
runtime fields, worker meaning, resume support and optional backend features.
This is the authority for dispatch; backend filenames are not a capability
contract.

## Named runtimes

Use `--runtime-config FILE --runtime NAME` when a converter belongs to another
Python environment. Relative `python` and `project_root` paths are resolved from
the config file, not from the caller's working directory. TOML and JSON are
accepted; `default_runtime` lets callers omit `--runtime`.

```toml
default_runtime = "pi05"

[runtimes.pi05]
python = "../openpi/.venv-pi05/bin/python"
project_root = "../openpi"

[runtimes.pi05.env]
XLA_PYTHON_CLIENT_PREALLOCATE = "false"
```

The older `--pi05-python`, `--lerobot-v3-python`, and `--openpi-root` options
remain supported. Supplying a conflicting legacy path and named-runtime path is
an error. Dry runs resolve and print named profiles without requiring paths to
exist on the planning machine; execution validates every declared path. Secret-
like environment values are redacted in printed and saved metadata.

When a child uses an external interpreter, the launcher prepends this project's
`src` directory to its inherited/profile `PYTHONPATH`. pi0.5 children also get
the selected OpenPI `src` and `packages/openpi-client/src` directories. The dry-
run `runtime.pythonpath_prepend` field shows these exact additions without
dumping the rest of the process environment.

## `--workers` contract

The unified option is intentionally mapped to each backend's real concurrency
control:

| Adapter | Meaning |
| --- | --- |
| PickPlace EgoSteer | Unused; this converter has no worker option. |
| Other EgoSteer adapters | Maximum parallel episode conversion processes. |
| EgoTouch | Maximum parallel episode-by-camera export processes. |
| LeRobot v3 | Parallel source preflight and episode conversion processes. |
| Fast pi0.5 wrappers | Source fingerprint/hash threads and fast episode conversion workers. |
| USB pi0.5 | Source fingerprint/hash threads and image-writer threads; image-writer processes remain disabled. |

Consult `--list-support` rather than assuming that the same number controls the
same resource for every format.

## Plans and conversion manifests

Every dry run emits `schema=kaihand_conversion_plan_v1`, `schema_version=1`,
and `stage=convert`, including selected runtime, worker semantics, source file
identity, and literal Raw action/camera contract fields. Contract values are
read from HDF5 attributes/groups; the entry point does not invent missing action
semantics. Historical episodes without `runtime_contract_json` are recorded as
`unknown_not_declared`, never filled from task defaults.

After a backend returns success, the entry point atomically publishes
`pipeline_conversion.json` in the output directory. This independent manifest
does not replace `dataset_manifest.json`, `conversion_manifest.json`, LeRobot
metadata, or any other backend-owned artifact. It records the successful plan,
the same Raw source identity and contracts, and SHA-256 references to recognized
backend manifests/source-identity files. A failed backend never publishes the
pipeline manifest. Existing `--resume` behavior is unchanged.
