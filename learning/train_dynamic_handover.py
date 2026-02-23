#!/usr/bin/env python3
# Copyright 2025 DeepMind Technologies Limited
#
# RL training script for DynamicHandover: bimanual Trossen arms bounce a
# ping pong ball between paddles. The policy observes arm qpos, qvel, ball
# (x,y,z), ball velocity, and time. Run with default PPO config:
#
#   uv run python learning/train_dynamic_handover.py
#
# Or override flags (e.g. shorter run, fewer envs):
#
#   uv run python learning/train_dynamic_handover.py --num_timesteps=5000000 --num_envs=512
#
# Full training (50M steps, 2048 envs):
#
#   uv run python learning/train_dynamic_handover.py --num_timesteps=50000000
#

import sys

# Inject --env_name=DynamicHandover so the generic trainer runs our env.
if not any(a.startswith('--env_name=') for a in sys.argv[1:]):
  sys.argv.insert(1, '--env_name=DynamicHandover')

from learning.train_jax_ppo import run

if __name__ == '__main__':
  run()
