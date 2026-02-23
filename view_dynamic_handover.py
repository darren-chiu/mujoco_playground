#!/usr/bin/env python3
"""Viewer for the DynamicHandover environment.

Use this to check:
  1. Ball physics – trajectory, bounces, gravity. Air drag (density/viscosity)
     is enabled in the viewer so the ball does not bounce infinitely.
  2. Collisions with paddles – ball should collide with left_paddle_geom and
     right_paddle_geom; you can move the arms in the viewer to test.
  3. Camera renderings – in the MuJoCo viewer, switch cameras via the
     "Camera" dropdown (cam_high, cam_low, cam_left_wrist, cam_right_wrist).

Note: The training env (MJX) uses the same XML but without fluid drag, because
MJX does not support implicitfast + fluid drag. So in the viewer you see
realistic ball decay; in training the ball can bounce longer.

Launch:
  uv run python view_dynamic_handover.py

Or with a specific initial ball velocity (e.g. toward the table):
  uv run python view_dynamic_handover.py --ball-vz -0.5
"""
import argparse
import pathlib

import mujoco
from mujoco import viewer
import numpy as np

# Path to the same XML used by the DynamicHandover env
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_XML_PATH = (
    _SCRIPT_DIR
    / "mujoco_playground"
    / "_src"
    / "manipulation"
    / "trossen_arm_mujoco"
    / "assets"
    / "dynamic_handover"
    / "dynamic_handover.xml"
)

# Arm start pose (matches env)
_START_ARM_POSE = [
    0.0,
    np.pi / 12,
    np.pi / 12,
    0.0,
    0.0,
    0.0,
    0.044,
    0.044,
    0.0,
    np.pi / 12,
    np.pi / 12,
    0.0,
    0.0,
    0.0,
    0.044,
    0.044,
]


def _parse_args():
  p = argparse.ArgumentParser(description="View DynamicHandover scene")
  p.add_argument(
      "--ball-vx",
      type=float,
      default=0.0,
      help="Initial ball velocity x (m/s)",
  )
  p.add_argument(
      "--ball-vy",
      type=float,
      default=0.0,
      help="Initial ball velocity y (m/s)",
  )
  p.add_argument(
      "--ball-vz",
      type=float,
      default=-0.3,
      help="Initial ball velocity z (m/s), e.g. -0.5 for downward",
  )
  p.add_argument(
      "--ball-x",
      type=float,
      default=0.0,
      help="Initial ball position x",
  )
  p.add_argument(
      "--ball-y",
      type=float,
      default=0.0,
      help="Initial ball position y",
  )
  p.add_argument(
      "--ball-z",
      type=float,
      default=0.25,
      help="Initial ball position z",
  )
  return p.parse_args()


# Air drag so the ball doesn't bounce infinitely (from ping_pong.xml).
# Only applied in the viewer; MJX training uses the XML as-is (no fluid).
_AIR_DENSITY = 1.2
_AIR_VISCOSITY = 1.8e-2


def load_callback(model=None, data=None):
  model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
  # Enable air drag in the viewer (standard MuJoCo supports this with implicitfast)
  model.opt.density = _AIR_DENSITY
  model.opt.viscosity = _AIR_VISCOSITY
  data = mujoco.MjData(model)

  # Arm: use keyframe "home" for qpos/ctrl, then override ball
  home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
  mujoco.mj_resetDataKeyframe(model, data, home_id)

  nq, nv = model.nq, model.nv
  # qpos layout: 16 arm, then 7 ball (3 pos + 4 quat)
  # qvel layout: 16 arm, then 6 ball (3 lin + 3 ang)
  arm_qpos_size = 16
  ball_qpos_start = 16
  ball_qvel_start = 16

  # Override arm to START_ARM_POSE (keyframe may already match)
  data.qpos[:arm_qpos_size] = np.array(_START_ARM_POSE, dtype=np.float64)

  # Ball initial position and quat (identity)
  data.qpos[ball_qpos_start : ball_qpos_start + 3] = [0.0, 0.0, 0.25]
  data.qpos[ball_qpos_start + 3 : ball_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]

  # Ball initial velocity (set from args in main)
  data.qvel[ball_qvel_start : ball_qvel_start + 3] = [0.0, 0.0, -0.3]
  data.qvel[ball_qvel_start + 3 : ball_qvel_start + 6] = [0.0, 0.0, 0.0]

  # Store for re-use in loader (viewer may call loader once)
  return model, data


def main():
  args = _parse_args()
  ball_pos = np.array([args.ball_x, args.ball_y, args.ball_z], dtype=np.float64)
  ball_vel = np.array([args.ball_vx, args.ball_vy, args.ball_vz], dtype=np.float64)

  def loader():
    model, data = load_callback()
    # Apply CLI ball state
    data.qpos[16:19] = ball_pos
    data.qpos[19:23] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[16:19] = ball_vel
    data.qvel[19:22] = [0.0, 0.0, 0.0]
    return model, data

  print("DynamicHandover viewer")
  print("  XML:", _XML_PATH)
  print("  Air drag: density={}, viscosity={} (ball will not bounce infinitely)".format(_AIR_DENSITY, _AIR_VISCOSITY))
  print("  Ball pos:", ball_pos)
  print("  Ball vel:", ball_vel)
  print("  Cameras: use viewer 'Camera' menu → cam_high, cam_low, cam_left_wrist, cam_right_wrist")
  print("  Paddles: left_paddle_geom, right_paddle_geom (ball should collide with them)")
  viewer.launch(loader=loader)


if __name__ == "__main__":
  main()
