# Conversion documentation

The canonical conversion guide is [docs/data_conversion.md](../../docs/data_conversion.md).

The model-neutral USB LeRobot v3 converter is documented in
[docs/usb_lerobot_v3.md](../../docs/usb_lerobot_v3.md).

The π0.5 dispatcher uses local compatibility wrappers for unified PickPlace,
Poker, Bulb, RAM, Vase and Whiteboard collections. The wrappers select only published
successes where a batch summary is authoritative, preserve outer collection
indices, and leave Raw files read-only. Their shared fast backend converts
episodes in separate processes, streams RGB directly to FFmpeg, publishes
video-backed LeRobot v2.1 without temporary PNG files, and resumes from atomic
episode checkpoints.

Run from the repository root:

```bash
pixi run convert-dataset -- --help
```
