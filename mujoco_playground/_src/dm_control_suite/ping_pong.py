# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ping pong ball environment (MJX-compatible).

Same physics scene as the repo-root ping_pong.xml, but without air drag so MJX
can load it. Use view_ping_pong.py + the root ping_pong.xml for viewing with
density/viscosity. This env is for having the ping pong ball model available in
the playground (e.g. debugging, or as a passive dynamics test).
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "ping_pong.xml"


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.0001,
      episode_length=2000,
      action_repeat=1,
      impl="jax",
      nconmax=1000,
      njmax=50,
  )


class PingPong(mjx_env.MjxEnv):
  """Ping pong ball bouncing in a box (floor + wall). No actuators; passive dynamics."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._xml_path = _XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._throw_key_id = mujoco.mj_name2id(
        self._mj_model, mujoco.mjtObj.mjOBJ_KEY, "throw"
    )

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Use CPU MjData to apply keyframe, then copy into MJX
    data_cpu = mujoco.MjData(self._mj_model)
    mujoco.mj_resetDataKeyframe(self._mj_model, data_cpu, self._throw_key_id)
    data = mjx.put_data(self._mj_model, data_cpu)
    data = mjx.forward(self._mjx_model, data)
    obs = jp.concatenate([data.qpos, data.qvel])
    info = {"rng": rng}
    return mjx_env.State(
        data, {"state": obs}, jp.zeros(()), jp.zeros(()), {}, info
    )

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # One dummy actuator (unused); pass zero ctrl
    ctrl = jp.zeros(self._mjx_model.nu)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    obs = jp.concatenate([data.qpos, data.qvel])
    done = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
    return mjx_env.State(
        data, {"state": obs}, jp.zeros(()), done.astype(jp.float32), state.metrics, state.info
    )

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
