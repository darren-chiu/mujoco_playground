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
"""DynamicHandover: bimanual Trossen arms bounce a ping pong ball between paddles."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

# Start pose for bimanual Trossen arms (matches trossen_arm_mujoco.constants.START_ARM_POSE)
_START_ARM_POSE = [
    0.0, np.pi / 12, np.pi / 12, 0.0, 0.0, 0.0, 0.044, 0.044,
    0.0, np.pi / 12, np.pi / 12, 0.0, 0.0, 0.0, 0.044, 0.044,
]


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.0005,  # Ball physics; 40 substeps per ctrl step
      episode_length=500,
      action_repeat=1,
      action_scale=0.02,
      ball_height_reward_scale=0.5,
      paddle_contact_reward_scale=2.0,
      alternate_bounce_bonus=1.0,
      impl='jax',
      nconmax=24 * 2048,
      njmax=88,
  )


class DynamicHandover(mjx_env.MjxEnv):
  """Bimanual Trossen arms with paddles must bounce a ping pong ball from one arm to the other."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    xml_path = (
        epath.Path(__file__).parent / 'assets' / 'dynamic_handover' /
        'dynamic_handover.xml'
    ).resolve().as_posix()
    self._xml_path = xml_path
    self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._post_init()

  def _post_init(self) -> None:
    self._ball_body_id = self._mj_model.body('ball').id
    self._ball_geom_id = self._mj_model.geom('ball').id
    self._left_paddle_geom_id = self._mj_model.geom('left_paddle_geom').id
    self._right_paddle_geom_id = self._mj_model.geom('right_paddle_geom').id

    home = self._mj_model.keyframe('home')
    self._init_q = jp.array(home.qpos)
    self._init_ctrl = jp.array(home.ctrl)
    self._lowers, self._uppers = np.array(self._mj_model.actuator_ctrlrange).T
    self._lowers = jp.array(self._lowers)
    self._uppers = jp.array(self._uppers)

    # qpos layout: 16 arm, then 7 for ball freejoint, then 4 for ballast ball joint
    self._arm_qpos_size = 16
    ball_jnt_id = self._mj_model.body_jntadr[self._ball_body_id]
    self._ball_qpos_adr = int(self._mj_model.jnt_qposadr[ball_jnt_id])
    self._ball_qvel_adr = int(self._mj_model.jnt_dofadr[ball_jnt_id])
    # Ballast (child of ball) has ball joint: +4 qpos, +3 qvel after the free joint
    self._ballast_qpos_adr = self._ball_qpos_adr + 7
    self._ballast_qvel_adr = self._ball_qvel_adr + 6

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

  def _ball_pos(self, data: mjx.Data) -> jax.Array:
    return data.xpos[self._ball_body_id]

  def _ball_vel(self, data: mjx.Data) -> jax.Array:
    return data.cvel[self._ball_body_id][3:6]  # linear velocity

  def _paddle_contacts(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    """Returns (left_paddle_contact, right_paddle_contact) as floats."""
    g1 = data.contact.geom[:, 0]
    g2 = data.contact.geom[:, 1]
    ball_id = jp.int32(self._ball_geom_id)
    left_id = jp.int32(self._left_paddle_geom_id)
    right_id = jp.int32(self._right_paddle_geom_id)
    left_contact = (
        ((g1 == ball_id) & (g2 == left_id)) | ((g1 == left_id) & (g2 == ball_id))
    ).any().astype(jp.float32)
    right_contact = (
        ((g1 == ball_id) & (g2 == right_id))
        | ((g1 == right_id) & (g2 == ball_id))
    ).any().astype(jp.float32)
    return left_contact, right_contact

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, arm_rng, which_arm_rng, ball_x_rng, ball_y_rng, ball_z_rng, ball_vel_rng = jax.random.split(rng, 7)

    # Arms at home pose
    start_pose = jp.array(_START_ARM_POSE)
    qpos = self._init_q.at[: self._arm_qpos_size].set(start_pose)

    # Spawn ball above one random arm's paddle (left y>0, right y<0) so it can hit
    which_arm = jax.random.randint(which_arm_rng, (), 0, 2)  # 0 = left, 1 = right
    # Left paddle region: x in [-0.08, 0.08], y in [0.25, 0.45]; right: y in [-0.45, -0.25]
    x_lo, x_hi = -0.08, 0.08
    left_y_lo, left_y_hi = 0.25, 0.45
    right_y_lo, right_y_hi = -0.45, -0.25
    ball_x = jax.random.uniform(ball_x_rng, (), minval=x_lo, maxval=x_hi)
    ball_y_left_rng, ball_y_right_rng = jax.random.split(ball_y_rng, 2)
    ball_y_left = jax.random.uniform(ball_y_left_rng, (), minval=left_y_lo, maxval=left_y_hi)
    ball_y_right = jax.random.uniform(ball_y_right_rng, (), minval=right_y_lo, maxval=right_y_hi)
    ball_y = jp.where(which_arm == 0, ball_y_left, ball_y_right)
    ball_z = 0.22 + 0.06 * jax.random.uniform(ball_z_rng, (), minval=0.0, maxval=1.0)
    ball_pos = jp.array([ball_x, ball_y, ball_z])
    ball_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    qpos = qpos.at[self._ball_qpos_adr : self._ball_qpos_adr + 3].set(ball_pos)
    qpos = qpos.at[self._ball_qpos_adr + 3 : self._ball_qpos_adr + 7].set(ball_quat)
    # Ballast quat (identity)
    qpos = qpos.at[self._ballast_qpos_adr : self._ballast_qpos_adr + 4].set(
        jp.array([1.0, 0.0, 0.0, 0.0])
    )

    # Small random velocity (slightly down or toward center)
    ball_vel_rng, vx_rng, vy_rng, vz_rng = jax.random.split(ball_vel_rng, 4)
    ball_vx = jax.random.uniform(vx_rng, (), minval=-0.15, maxval=0.15)
    ball_vy_left_rng, ball_vy_right_rng = jax.random.split(vy_rng, 2)
    ball_vy_left = jax.random.uniform(ball_vy_left_rng, (), minval=-0.2, maxval=0.0)
    ball_vy_right = jax.random.uniform(ball_vy_right_rng, (), minval=0.0, maxval=0.2)
    ball_vy = jp.where(which_arm == 0, ball_vy_left, ball_vy_right)  # toward center
    ball_vz = -0.05 + 0.1 * jax.random.uniform(vz_rng, (), minval=0.0, maxval=1.0)
    qvel = jp.zeros(self._mjx_model.nv)
    qvel = qvel.at[self._ball_qvel_adr].set(ball_vx)
    qvel = qvel.at[self._ball_qvel_adr + 1].set(ball_vy)
    qvel = qvel.at[self._ball_qvel_adr + 2].set(ball_vz)
    # Ball and ballast angular vel zero
    qvel = qvel.at[self._ball_qvel_adr + 3 : self._ballast_qvel_adr + 3].set(0.0)

    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    info = {
        'rng': rng,
        'last_left_contact': jp.array(0.0),
        'last_right_contact': jp.array(0.0),
        'bounce_count': jp.array(0.0),
        '_steps': jp.array(0, dtype=int),
    }
    obs = self._get_obs(data, info)
    reward = jp.array(0.0)
    done = jp.array(0.0)
    metrics = {
        'ball_height': jp.array(0.0),
        'left_contact': jp.array(0.0),
        'right_contact': jp.array(0.0),
        'bounce_count': jp.array(0.0),
    }
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    newly_reset = state.info['_steps'] == 0
    state.info['last_left_contact'] = jp.where(
        newly_reset, 0.0, state.info['last_left_contact']
    )
    state.info['last_right_contact'] = jp.where(
        newly_reset, 0.0, state.info['last_right_contact']
    )

    delta = action * self._config.action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(
        self._mjx_model, state.data, ctrl, self.n_substeps
    )

    ball_pos = self._ball_pos(data)
    left_contact, right_contact = self._paddle_contacts(data)

    # Reward: keep ball up + reward paddle contacts (alternating is ideal)
    height_reward = self._config.ball_height_reward_scale * jp.maximum(
        ball_pos[2] - 0.05, 0.0
    )
    contact_reward = (
        self._config.paddle_contact_reward_scale * (left_contact + right_contact)
    )
    # Bonus for alternating: hit right after left or left after right
    alternate = (
        left_contact * state.info['last_right_contact']
        + right_contact * state.info['last_left_contact']
    )
    alternate_bonus = self._config.alternate_bounce_bonus * alternate
    reward = (height_reward + contact_reward + alternate_bonus) * self.dt

    state.info['last_left_contact'] = left_contact
    state.info['last_right_contact'] = right_contact
    state.info['bounce_count'] = state.info['bounce_count'] + left_contact + right_contact

    steps_after = state.info['_steps'] + self._config.action_repeat
    time_limit_reached = steps_after >= self._config.episode_length
    done = (
        (ball_pos[2] < 0.02)
        | jp.any(jp.abs(ball_pos) > 1.0)
        | jp.any(jp.isnan(data.qpos))
        | jp.any(jp.isnan(data.qvel))
        | time_limit_reached
    )
    state.info['_steps'] = state.info['_steps'] + self._config.action_repeat
    state.info['_steps'] = jp.where(
        done | (state.info['_steps'] >= self._config.episode_length),
        0,
        state.info['_steps'],
    )

    state.metrics.update(
        ball_height=ball_pos[2],
        left_contact=left_contact,
        right_contact=right_contact,
        bounce_count=state.info['bounce_count'],
    )

    obs = self._get_obs(data, state.info)
    return mjx_env.State(
        data, obs, reward, done.astype(jp.float32), state.metrics, state.info
    )

  def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
    arm_qpos = data.qpos[: self._arm_qpos_size]
    arm_qvel = data.qvel[: self._arm_qpos_size]
    ball_xyz = self._ball_pos(data)
    ball_vel = self._ball_vel(data)
    time_frac = (info['_steps'].astype(jp.float32) /
                  jp.maximum(1, self._config.episode_length))

    state = jp.concatenate([
        arm_qpos,
        arm_qvel,
        ball_xyz,
        ball_vel,
        jp.array([time_frac]),
    ])
    return {'state': state}
