#!/usr/bin/env python3
"""Evaluate one frozen FTP-1 checkpoint with its native shared task monitor."""
from evaluate_poker_tactile_policy_batch import install_termination_handler, main

if __name__ == "__main__":
  install_termination_handler()
  raise SystemExit(main(shared_tasks=True))
