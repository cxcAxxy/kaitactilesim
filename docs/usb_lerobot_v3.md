# USB raw to canonical LeRobot v3

The converter is `scripts/workcell/convert_usb_to_lerobot_v3.py`. It targets
the shared dataset used by the six non-pi0.5 models; it does not replace the
separate pi0.5 conversion.

## Run

Use the environment that currently provides LeRobot 0.4.2 / codebase v3.0:

```bash
env HF_HOME=/cpfs_infra/user/chenxianchi/.cache/huggingface \
  /cpfs_infra/user/chenxianchi/miniconda3/envs/lingbot_vla2/bin/python \
  scripts/workcell/convert_usb_to_lerobot_v3.py \
  --input-dir /nas/chenxianchi/datasets/sim/usb_insert/raw \
  --output-dir /nas/chenxianchi/datasets/sim/usb_insert/lerobot_v3/0920_200 \
  --repo-id kaihand/usb_insert_0920_200 \
  --expected-episodes 200 \
  --workers 3 \
  --ffmpeg-threads 2 \
  --resume
```

The input may be either the `raw` parent shown above or the concrete
`raw/0920_200` collection. The output path must not already exist. The default
work directory is
`/nas/chenxianchi/datasets/sim/usb_insert/lerobot_v3/.0920_200.resume-work`.
Set `--work-dir PATH` to choose it explicitly. Keep the work directory on the
same filesystem as the final output when possible, so final assembly can use
hard links instead of making another full copy.

The command above runs three episode workers. Each worker uses two x264
threads and encodes the head and wrist videos sequentially, so at most three
FFmpeg encoders (about six x264 threads) run at once. Increase `--workers` only
after checking NAS throughput and memory, not just free CPU cores.

Each worker writes one episode Parquet and one MP4 per camera into a private
directory, calls `fsync`, writes `done.json`, and atomically commits that
episode. RGB batches are piped directly from HDF5 to FFmpeg; no temporary PNG
files are created. LeRobot-compatible per-channel image statistics are sampled
during that same read. The final result remains a standard LeRobot v3 layout.

## Resume after interruption

Run the exact same command again. `--resume` performs two fast checks:

- the saved collection/control-file hashes and every raw file's size/mtime
  must still match;
- a committed episode's `done.json` and artifact sizes must match its saved
  plan.

It does not repeat the full HDF5 structural preflight and does not reopen or
re-encode completed episodes. A log such as
`resume: completed=83 pending=117` shows the recovery point. Partial episode
directories left by a power loss are discarded and regenerated. The work
directory is removed after successful publication; pass `--keep-work-dir` if
an audit copy of the episode checkpoints is wanted.

For a read-only source check:

```bash
/cpfs_infra/user/chenxianchi/miniconda3/envs/lingbot_vla2/bin/python \
  scripts/workcell/convert_usb_to_lerobot_v3.py \
  --input-dir /nas/chenxianchi/datasets/sim/usb_insert/raw \
  --output-dir /tmp/not-written \
  --expected-episodes 200 \
  --validate-only
```

Use `--verify-source-hash` when an independent full reread of all HDF5 bytes is
needed. Without it, the converter trusts the acquisition-time SHA-256 in each
paired JSON sidecar and still validates the HDF5 structure and values.

## Clock and action contract

- Dataset FPS is 30, driven by the head/right-wrist camera clock.
- State, wrench, tactile and diagnostics use the recorded latest-nonfuture
  500 Hz state index for each camera frame.
- An action is the next retained camera waypoint. Therefore the last retained
  camera frame is excluded.
- `action` is 27D: next actual right-wrist world pose
  `[x,y,z,qx,qy,qz,qw]` plus the next 20D right-hand controller target.
- `auxiliary.action.right_joint_target` is the alternative 27D controller view:
  7D right-arm target plus 20D right-hand target.

## Main fields

The frame table contains:

- `observation.images.head`, `observation.images.right_wrist`
- `observation.state` (current actual wrist pose + 20 actuated hand joints)
- `observation.state.right_joint_position` (actual 7 arm + 20 hand joints)
- split arm/hand position, velocity and effort fields
- actual wrist pose and five fingertip poses
- local and world right-wrist 6D wrench
- right tactile taxel force `(5,7,5,3)` plus aggregate contact/force fields
- Genesis tactile probe fields under `auxiliary.tactile_genesis.right.*`
- main wrist/hand action and alternative arm/hand joint targets
- primary-object pose/twist and USB diagnostics under `auxiliary.*`
- exact raw episode/frame/state/timestamp provenance

`meta/kaihand_schema.json` is the machine-readable authority for coordinate,
quaternion, action, tactile and phase semantics. `meta/kaihand_source_episodes.jsonl`
maps every output episode back to the immutable raw HDF5 and its SHA-256.

The quaternion part of wrist pose is absolute XYZW. Never create a relative
action by component-wise `action - observation.state`; an adapter that needs
relative end-effector actions must compute an SE(3) delta (for example
translation plus axis-angle).

The left robot/camera, 500 Hz full physics arrays, ragged contact events and
sparse controller diagnostics remain in raw. They are intentionally not
duplicated or falsely represented as 30 Hz frame data. Object state, task
phase and success are privileged simulation diagnostics and should not be fed
to a deployable policy unless the same signals exist at inference time.
