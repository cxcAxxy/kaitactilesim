# Unified collection command

The canonical collection guide is [docs/workflows/data_collection.md](../../docs/workflows/data_collection.md).

The public collector covers all eleven registered tasks. Inspect the supported
task profiles without loading MuJoCo or starting a recorder:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run collect -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run collect -- --describe vase-wipe
```

Start a production collection from the repository root:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
  /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run collect -- \
  --task vase-wipe \
  --profile production-v1 \
  --render-backend hardware \
  --output-dir /nas/chenxianchi/datasets/vase-wipe-v1
```

The vendor JSON only selects the host NVIDIA EGL implementation for hardware
rendering. It is not a collection profile or a model runtime configuration;
the recorder still writes the actual physics and tactile `runtime_contract_json`.

`production-v1` is the default. `legacy-compatible` is an exact alias that
keeps the pre-profile recorder command, argument order, cameras, camera rate,
randomization, and task-owned outcome gate unchanged. `--fixed-stains` still
only affects VaseWipe stain seeds and WhiteboardWipe ink seeds.

Each new `collection.json` records the canonical profile, the fully resolved
per-task profile, and the concrete first production command (including its
derived seed and staging path). Every `status.json` records the exact command
for that attempt. A successful status also copies the recorder-authored
`runtime_contract_json` from each HDF5 root; an older HDF5 without that
attribute is reported as `unknown`, never filled with inferred physics or
tactile values.

Legacy-shape `collection.json` files remain readable by `--resume` only when
their complete old contract, including `source_sha256`, still matches. Adding
or changing source files therefore still requires a new collection directory.

For all options:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run collect -- --help
```
