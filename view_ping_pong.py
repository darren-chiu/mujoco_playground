#!/usr/bin/env python3
"""Quick viewer for the bouncing ping pong MuJoCo model."""
import pathlib

import mujoco
from mujoco import viewer

_XML_PATH = pathlib.Path(__file__).resolve().parent / "ping_pong.xml"


def load_callback(model=None, data=None):
  model = mujoco.MjModel.from_xml_path(_XML_PATH.as_posix())
  data = mujoco.MjData(model)
  # Apply "throw" keyframe so the ball starts with initial velocity
  throw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "throw")
  mujoco.mj_resetDataKeyframe(model, data, throw_id)
  return model, data


if __name__ == "__main__":
  viewer.launch(loader=load_callback)
