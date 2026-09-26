import pytest
from kaihand_tactile_env.shared.evaluation_resume import require_valid_resume_trials


def test_resume_accepts_only_completed_valid_trials() -> None:
  require_valid_resume_trials([
    {"seed": 0, "status": "success", "success": True,
     "valid_trial": True, "returncode": 0},
    {"seed": 1, "status": "task_not_completed", "success": False,
     "valid_trial": True, "returncode": 0},
  ])


@pytest.mark.parametrize("change", [
  {"status": "error", "success": False, "valid_trial": False},
  {"returncode": 17},
  {"success": False},
])
def test_resume_rejects_failed_trial_row(change: dict) -> None:
  row = {"seed": 0, "status": "success", "success": True,
         "valid_trial": True, "returncode": 0}
  row.update(change)
  with pytest.raises(RuntimeError, match="invalid or incomplete"):
    require_valid_resume_trials([row])
