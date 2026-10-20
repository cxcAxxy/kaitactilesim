"""Task-scoped tactile contracts; registration is not checkpoint availability.

Instructions below describe the exported training corpus, not a relabelling of
that corpus to the simulator's newer completion protocol. All new deployments
remain diagnostic until their own closed-loop acceptance is completed.
"""
from __future__ import annotations

import json
from pathlib import Path

POKER_INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the "
  "fingers and thumb, lift it, and turn its face toward the robot to look at it."
)

_NONPOKER_ACTION_SUPERVISION_CONTRACT = (
  "ftp1-native-future-measured-state-trajectory-v1"
)


def _contract(domain, instruction, body, *, flex=False):
  return dict(domain_name=domain, instruction=instruction, tactile_target_body=body,
              tactile_target_geometries="enabled_collision_geometries_on_target_body",
              tactile_provider="native_sponge_flex" if flex else "solver_distributed_rigid")


FTP1_TASK_CONTRACTS = {
  "poker-draw": _contract("KaiHandSimPokerDraw", POKER_INSTRUCTION, "card"),
  "bulb-screw": _contract("KaiHandSimBulbScrew",
    "Grasp the light bulb with the right hand, lift it to the socket, screw it in clockwise, release it, and withdraw the hand.", "bulb"),
  "bulb-screw-inplace": _contract("KaiHandSimBulbScrewInplace",
    "Grasp the light bulb already on the socket with the right hand, screw it in clockwise until seated, release it, and withdraw the hand.", "bulb"),
  "bulb-unscrew": _contract("KaiHandSimBulbUnscrew",
    "Grasp the installed light bulb with the right hand, unscrew it, return it to the table, release it, and withdraw the hand.", "bulb"),
  "install-ram": _contract("KaiHandSimInstallRam",
    "Grasp the RAM module with the right hand, align it with the memory slot, insert it fully, release it, and withdraw the hand.", "ram"),
  "usb-insert": _contract("KaiHandSimUsbInsert",
    "Grasp the USB plug with the right hand, align it with the socket, insert it fully, release it, and withdraw the hand.", "usb_plug"),
  "usb-unplug": _contract("KaiHandSimUsbUnplug",
    "Grasp the already inserted USB plug with the right hand, pull it straight out of the socket, return it to the table, release it, and withdraw the hand.", "usb_plug"),
  "whiteboard-wipe": _contract("KaiHandSimWhiteboardWipe",
    "Grasp the eraser with the right hand, wipe the marks off the inclined whiteboard, return the eraser to the table, release it, and withdraw the hand.", "eraser"),
  "whiteboard-wipe-horizontal": _contract("KaiHandSimWhiteboardWipeHorizontal",
    "Grasp the eraser with the right hand, wipe the marks off the horizontal whiteboard, lift the eraser, return it to the table, release it, and withdraw the hand.", "eraser"),
  "whiteboard-wipe-horizontal-line": _contract("KaiHandSimWhiteboardWipeHorizontalLine",
    "Grasp the eraser with the right hand, wipe the single horizontal line off the horizontal whiteboard, lift the eraser, return it to the table, release it, and withdraw the hand.", "eraser"),
  "vase-wipe": _contract("KaiHandSimVaseWipe",
    "Pick up the sponge with the right hand and wipe the marked dirt from the inside wall of the vase until it is clean.", "sponge", flex=True),
  "sponge-grasp": _contract("KaiHandSimSpongeGrasp",
    "Pick up the upright sponge with the right hand, carry it to the plate on the robot's right, place it inside the plate, and release it.", "sponge", flex=True),
}
SHARED_TACTILE_TASKS = tuple(task for task in FTP1_TASK_CONTRACTS if task != "poker-draw")


def task_contract(task):
  try:
    return FTP1_TASK_CONTRACTS[task]
  except KeyError as error:
    raise ValueError(f"tactile closed-loop adapter is not implemented for {task!r}") from error


def ftp1_training_binding(checkpoint, task):
  """Check and return the task/domain/corpus identity without loading weights.

  A copied/renamed normalization directory is insufficient: the checkpoint's
  training dataset config and its conversion provenance must agree as well.
  Hashing these returned paths belongs to deployment preparation.
  """
  checkpoint = Path(checkpoint)
  contract = task_contract(task)
  train = json.loads((checkpoint / "train_config.json").read_text())
  config_path = Path(train["dataset_config_path"]).expanduser().resolve(strict=True)
  config = json.loads(config_path.read_text())
  datasets = [row for row in config.get("datasets", [])
              if row.get("enabled", True) and row.get("name") == contract["domain_name"]]
  if len(datasets) != 1:
    raise ValueError(f"checkpoint training dataset does not uniquely bind {task} / {contract['domain_name']}")
  corpus = Path(datasets[0]["path"]).expanduser().resolve(strict=True)
  conversion_path = corpus / "conversion_manifest.json"
  conversion = json.loads(conversion_path.read_text())
  if conversion.get("task") != task or conversion.get("instruction") != contract["instruction"]:
    raise ValueError(f"checkpoint training corpus task/instruction differs from {task}")
  if conversion.get("fps") != 30 or conversion.get("ftp1_action_horizon") != 32:
    raise ValueError("FTP-1 training corpus must use the native 30 Hz / H32 clock")
  action_supervision = conversion.get("action_supervision")
  if task != "poker-draw":
    if not isinstance(action_supervision, dict):
      raise ValueError(
        "non-poker FTP-1 conversion action_supervision must be an object"
      )
    if (
      action_supervision.get("contract")
      != _NONPOKER_ACTION_SUPERVISION_CONTRACT
      or action_supervision.get("controller_targets_used") is not False
      or action_supervision.get("pi05_recorded_controller_target_equivalent")
      is not False
    ):
      raise ValueError(
        "non-poker FTP-1 conversion action_supervision contract mismatch"
      )
  recipe = train.get("action_joint_rep")
  if recipe not in {"mix", "arm_relative_hand_absolute"}:
    raise ValueError(f"unsupported FTP-1 training action_joint_rep: {recipe!r}")
  return dict(dataset_config_path=str(config_path), conversion_manifest_path=str(conversion_path),
              dataset_root=str(corpus), domain_name=contract["domain_name"], task=task,
              instruction=contract["instruction"], action_joint_rep=recipe,
              action_supervision=action_supervision)
