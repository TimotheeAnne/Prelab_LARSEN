"""This file implements the gym environment of minitaur.

"""
import math
import time

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from fast_adaptation_embedding.env.assets.pybullet_envs import bullet_client
from fast_adaptation_embedding.env.assets import pybullet_data
from fast_adaptation_embedding.env.assets.pybullet_envs import minitaur
from fast_adaptation_embedding.env.assets.pybullet_envs import minitaur_logging
from fast_adaptation_embedding.env.assets.pybullet_envs import minitaur_logging_pb2
from fast_adaptation_embedding.env.assets.pybullet_envs import motor
from pkg_resources import parse_version

NUM_MOTORS = 8
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 360
RENDER_WIDTH = 480
SENSOR_NOISE_STDDEV = minitaur.SENSOR_NOISE_STDDEV
DEFAULT_URDF_VERSION = "default"
DERPY_V0_URDF_VERSION = "derpy_v0"
RAINBOW_DASH_V0_URDF_VERSION = "rainbow_dash_v0"
NUM_SIMULATION_ITERATION_STEPS = 300

MINIATUR_URDF_VERSION_MAP = {
    DEFAULT_URDF_VERSION: minitaur.Minitaur,
    # DERPY_V0_URDF_VERSION: minitaur_derpy.MinitaurDerpy,
    # RAINBOW_DASH_V0_URDF_VERSION: minitaur_rainbow_dash.MinitaurRainbowDash,
}


def convert_to_list(obj):
  try:
    iter(obj)
    return obj
  except TypeError:
    return [obj]


class MinitaurControlledEnv(gym.Env):
  """The gym environment for the minitaur.

  It simulates the locomotion of a minitaur, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the minitaur walks in 1000 steps and penalizes the energy
  expenditure.

  """
  metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 25}

  def __init__(self,
               urdf_root=pybullet_data.getDataPath(),
               urdf_version=None,
               distance_weight=1.0,
               energy_weight=0.0,
               shake_weight=0.0,
               drift_weight=0.0,
               survival_weight=0.0,
               distance_limit=float("inf"),
               observation_noise_stdev=SENSOR_NOISE_STDDEV,
               self_collision_enabled=True,
               motor_velocity_limit=np.inf,
               pd_control_enabled=False,
               leg_model_enabled=True,
               accurate_motor_model_enabled=False,
               remove_default_joint_damping=False,
               motor_kp=1.0,
               motor_kd=0.02,
               control_latency=0.0,
               pd_latency=0.0,
               torque_control_enabled=False,
               motor_overheat_protection=False,
               hard_reset=True,
               on_rack=False,
               render=False,
               num_steps_to_log=1000,
               action_repeat=1,
               control_time_step=None,
               env_randomizer=None,
               forward_reward_cap=float("inf"),
               reflection=True,
               log_path=None,
               ):
    """Initialize the minitaur gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION,
        RAINBOW_DASH_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. DERPY_V0_URDF_VERSION
        is the result of first pass system identification for derpy.
        We will have a different URDF and related Minitaur class each time we
        perform system identification. While the majority of the code of the
        class remains the same, some code changes (e.g. the constraint location
        might change). __init__() will choose the right Minitaur class from
        different minitaur modules based on
        urdf_version.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      remove_default_joint_damping: Whether to remove the default joint damping.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      control_latency: It is the delay in the controller between when an
        observation is made at some point, and when that reading is reported
        back to the Neural Network.
      pd_latency: latency of the PD controller loop. PD calculates PWM based on
        the motor angle and velocity. The latency measures the time between when
        the motor angle and velocity are observed on the microcontroller and
        when the true state happens on the motor. It is typically (0.001-
        0.002s).
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the minitaur back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode that will
        be logged. If the number of steps is more than num_steps_to_log, the
        environment will still be running, but only first num_steps_to_log will
        be recorded in logging.
      action_repeat: The number of simulation steps before actions are applied.
      control_time_step: The time step between two successive control signals.
      env_randomizer: An instance (or a list) of EnvRandomizer(s). An
        EnvRandomizer may randomize the physical property of minitaur, change
          the terrrain during reset(), or add perturbation forces during step().
      forward_reward_cap: The maximum value that forward reward is capped at.
        Disabled (Inf) by default.
      log_path: The path to write out logs. For the details of logging, refer to
        minitaur_logging.proto.
    Raises:
      ValueError: If the urdf_version is not supported.
    """
    # Set up logging.
    self._log_path = log_path
    self.logging = minitaur_logging.MinitaurLogging(log_path)
    # PD control needs smaller time step for stability.
    if control_time_step is not None:
      self.control_time_step = control_time_step
      self._action_repeat = action_repeat
      self._time_step = control_time_step / action_repeat
    else:
      # Default values for time step and action repeat
      if accurate_motor_model_enabled or pd_control_enabled:
        self._time_step = 0.002
        self._action_repeat = 5
      else:
        self._time_step = 0.01
        self._action_repeat = 1
      self.control_time_step = self._time_step * self._action_repeat
    # TODO(b/73829334): Fix the value of self._num_bullet_solver_iterations.
    self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
    self._urdf_root = urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_velocity_limit = motor_velocity_limit
    self._observation = []
    self._true_observation = []
    self._objectives = []
    self._objective_weights = [distance_weight, energy_weight, drift_weight, shake_weight, survival_weight]
    self._env_step_counter = 0
    self._num_steps_to_log = num_steps_to_log
    self._is_render = render
    self._last_base_position = [0, 0, 0]
    self._distance_weight = distance_weight
    self._energy_weight = energy_weight
    self._drift_weight = drift_weight
    self._shake_weight = shake_weight
    self._distance_limit = distance_limit
    self._observation_noise_stdev = observation_noise_stdev
    self._action_bound = 1
    self._pd_control_enabled = pd_control_enabled
    self._leg_model_enabled = leg_model_enabled
    self._accurate_motor_model_enabled = accurate_motor_model_enabled
    self._remove_default_joint_damping = remove_default_joint_damping
    self._motor_kp = motor_kp
    self._motor_kd = motor_kd
    self._torque_control_enabled = torque_control_enabled
    self._motor_overheat_protection = motor_overheat_protection
    self._on_rack = on_rack
    self._cam_dist = 1.0
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._forward_reward_cap = forward_reward_cap
    self._hard_reset = True
    self._last_frame_time = 0.0
    self._control_latency = control_latency
    self._pd_latency = pd_latency
    self._urdf_version = urdf_version
    self._ground_id = None
    self._reflection = reflection
    self._env_randomizers = convert_to_list(env_randomizer) if env_randomizer else []
    self._episode_proto = minitaur_logging_pb2.MinitaurEpisode()
    self._slope_degree = 0
    self._friction = 1
    self._unblocked_steering = True
    self.controller = self.controller_sawtooth
    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bullet_client.BulletClient()
    if self._urdf_version is None:
      self._urdf_version = DEFAULT_URDF_VERSION
    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self.seed()
    self.reset()
    observation_high = (self._get_observation_upper_bound() + OBSERVATION_EPS)
    observation_low = (self._get_observation_lower_bound() - OBSERVATION_EPS)
    action_dim = 5  # [steering, step_size, leg_extension, leg_extension_offset]
    action_high = np.array([self._action_bound] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(observation_low, observation_high)
    self.viewer = None
    self._hard_reset = hard_reset  # This assignment need to be after reset()

  def steering(self, unblock):
      self._unblocked_steering = unblock

  def close(self):
    if self._env_step_counter > 0:
      self.logging.save_episode(self._episode_proto)
    self.minitaur.Terminate()

  def add_env_randomizer(self, env_randomizer):
    self._env_randomizers.append(env_randomizer)

  def reset(self, initial_motor_angles=None, reset_duration=1.0):
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
    if self._env_step_counter > 0:
      self.logging.save_episode(self._episode_proto)
    self._episode_proto = minitaur_logging_pb2.MinitaurEpisode()
    minitaur_logging.preallocate_episode_proto(self._episode_proto, self._num_steps_to_log)
    if self._hard_reset:
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
      self._pybullet_client.changeDynamics(self._ground_id, linkIndex=-1, lateralFriction=self._friction)
      if (self._reflection):
        self._pybullet_client.changeVisualShape(self._ground_id, -1, rgbaColor=[1, 1, 1, 0.8])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, self._ground_id)
      alpha = np.pi/180*self._slope_degree
      self._pybullet_client.setGravity(-10*np.sin(alpha), 0, -10*np.cos(alpha))
      acc_motor = self._accurate_motor_model_enabled
      motor_protect = self._motor_overheat_protection
      if self._urdf_version not in MINIATUR_URDF_VERSION_MAP:
        raise ValueError("%s is not a supported urdf_version." % self._urdf_version)
      else:
        self.minitaur = (MINIATUR_URDF_VERSION_MAP[self._urdf_version](
            pybullet_client=self._pybullet_client,
            action_repeat=self._action_repeat,
            urdf_root=self._urdf_root,
            time_step=self._time_step,
            self_collision_enabled=self._self_collision_enabled,
            motor_velocity_limit=self._motor_velocity_limit,
            pd_control_enabled=self._pd_control_enabled,
            accurate_motor_model_enabled=acc_motor,
            remove_default_joint_damping=self._remove_default_joint_damping,
            motor_kp=self._motor_kp,
            motor_kd=self._motor_kd,
            control_latency=self._control_latency,
            pd_latency=self._pd_latency,
            observation_noise_stdev=self._observation_noise_stdev,
            torque_control_enabled=self._torque_control_enabled,
            motor_overheat_protection=motor_protect,
            on_rack=self._on_rack))
    self.minitaur.Reset(reload_urdf=False,
                        default_motor_angles=initial_motor_angles,
                        reset_time=reset_duration)
    # self.minitaur.SetFootFriction(self._friction)
    # Loop over all env randomizers.
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_env(self)

    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self._env_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._objectives = []
    self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                     self._cam_pitch, [0, 0, 0])
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
    return self._get_observation()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _transform_action_to_motor_command(self, action):
    if self._leg_model_enabled:
      for i, action_component in enumerate(action):
        if not (-self._action_bound - ACTION_EPS <= action_component <=
                self._action_bound + ACTION_EPS):
          raise ValueError("{}th action {} out of bounds.".format(i, action_component))
      action = self.minitaur.ConvertFromLegModel(action)
    return action

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    self._last_base_position = self.minitaur.GetBasePosition()

    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self.control_time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
      base_pos = self.minitaur.GetBasePosition()
      # Keep the previous orientation of the camera set by the user.
      [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
      self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_step(self)
    t = self.minitaur.GetTimeSinceReset()
    action = self.controller(action, t)
    action = self._transform_action_to_motor_command(action)
    self.minitaur.Step(action)
    reward = self._reward()
    done = self._termination()
    if self._log_path is not None:
      minitaur_logging.update_episode_proto(self._episode_proto, self.minitaur, action,
                                            self._env_step_counter)
    self._env_step_counter += 1
    if done:
      self.minitaur.Terminate()
    return np.array(self._get_observation()), reward, done, {'action': action, 'rewards': self._objectives}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.minitaur.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=-self._cam_dist,
        yaw=self._cam_yaw + self._slope_degree,
        pitch=-80,
        roll=0,
        upAxisIndex=1)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                   aspect=float(RENDER_WIDTH) /
                                                                   RENDER_HEIGHT,
                                                                   nearVal=0.1,
                                                                   farVal=100.0)
    (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def set_mismatch(self, mismatch):
      slope = mismatch[0]
      friction = mismatch[1]
      self._slope_degree = 25 * slope
      self._friction = (1 + 9 * friction) if friction >= 0 else (1 + friction)

  def get_foot_contact(self):
      """
      FR r3 - l6
      BR r9 - l12
      FL l16 -r19
      BL l22 - r25
      :return: [FR, BR, FL, BL] boolean contact with the floor
      """
      contacts = []
      for foot_id in self.minitaur._foot_link_ids:
        contact = self.pybullet_client.getContactPoints(0, 1, -1, foot_id)
        if contact != ():
            contacts.append(foot_id)
      FR = 3 in contacts or 6 in contacts
      BR = 9 in contacts or 12 in contacts
      FL = 16 in contacts or 19 in contacts
      BL = 22 in contacts or 25 in contacts
      return [FR, BR, FL, BL]

  def get_foot_position(self):
      """
      FR r3 - l6
      BR r9 - l12
      FL l16 -r19
      BL l22 - r25
      :return: [FR, BR, FL, BL] boolean contact with the floor
      """
      Positions = []
      for foot_id in self.minitaur._foot_link_ids:
        pos = self.pybullet_client.getLinkState(self.minitaur.quadruped, foot_id)
        if foot_id in [3, 6, 9, 12, 16, 19, 22, 25]:
            Positions.append(pos[0])
      return Positions

  def get_minitaur_motor_angles(self):
    """Get the minitaur's motor angles.

    Returns:
      A numpy array of motor angles.
    """
    return np.array(self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:MOTOR_ANGLE_OBSERVATION_INDEX +
                                      NUM_MOTORS])

  def get_minitaur_motor_velocities(self):
    """Get the minitaur's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
    return np.array(
        self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:MOTOR_VELOCITY_OBSERVATION_INDEX +
                          NUM_MOTORS])

  def get_minitaur_motor_torques(self):
    """Get the minitaur's motor torques.

    Returns:
      A numpy array of motor torques.
    """
    return np.array(
        self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:MOTOR_TORQUE_OBSERVATION_INDEX +
                          NUM_MOTORS])

  def get_minitaur_base_orientation(self):
    """Get the minitaur's base orientation, represented by a quaternion.

    Returns:
      A numpy array of minitaur's orientation.
    """
    return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

  def is_fallen(self):
    """Decide whether the minitaur has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the minitaur is considered fallen.

    Returns:
      Boolean value that indicates whether the minitaur has fallen.
    """
    # orientation = self.minitaur.GetBaseOrientation()
    # rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    # local_up = rot_mat[6:]
    pos = self.minitaur.GetBasePosition()
    return pos[2] < 0.13

  def _termination(self):
    position = self.minitaur.GetBasePosition()
    distance = math.sqrt(position[0]**2 + position[1]**2)
    return self.is_fallen() or distance > self._distance_limit

  def _reward(self):
    current_base_position = self.minitaur.GetBasePosition()
    forward_reward = current_base_position[0] - self._last_base_position[0]
    # Cap the forward reward if a cap is set.
    forward_reward = min(forward_reward, self._forward_reward_cap)
    # Penalty for sideways translation.
    drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
    # Penalty for sideways rotation of the body.
    orientation = self.minitaur.GetBaseOrientation()
    rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
    local_up_vec = rot_matrix[6:]
    shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
    survival_reward = -(current_base_position[2] < 0.13)
    energy_reward = -np.abs(
        np.dot(self.minitaur.GetMotorTorques(),
               self.minitaur.GetMotorVelocities())) * self._time_step
    objectives = [forward_reward, energy_reward, drift_reward, shake_reward, survival_reward]
    weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
    reward = sum(weighted_objectives)
    self._objectives.append(objectives)
    return reward

  def get_objectives(self):
    return self._objectives

  @property
  def objective_weights(self):
    """Accessor for the weights for all the objectives.

    Returns:
      List of floating points that corresponds to weights for the objectives in
      the order that objectives are stored.
    """
    return self._objective_weights

  def _get_observation(self):
    """Get observation of this environment, including noise and latency.

    The minitaur class maintains a history of true observations. Based on the
    latency, this function will find the observation at the right time,
    interpolate if necessary. Then Gaussian noise is added to this observation
    based on self.observation_noise_stdev.

    Returns:
      The noisy observation with latency.
    """
    observation = []
    observation.extend(list(self.minitaur.GetBasePosition())[:2])
    quaternion = self.minitaur.GetBaseOrientation()
    euler = pybullet.getEulerFromQuaternion(quaternion)
    yaw = euler[2]
    z_ang_sin = np.sin(yaw)
    z_ang_cos = np.cos(yaw)
    observation.extend([z_ang_sin, z_ang_cos])
    # observation.extend(self.get_foot_contact())
    self._observation = observation
    return self._observation

  def _get_true_observation(self):
    """Get the observations of this environment.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
    observation = []
    observation.extend(self.minitaur.GetTrueMotorAngles().tolist())
    observation.extend(self.minitaur.GetTrueMotorVelocities().tolist())
    observation.extend(self.minitaur.GetTrueMotorTorques().tolist())
    observation.extend(list(self.minitaur.GetTrueBaseOrientation()))

    self._true_observation = observation
    return self._true_observation

  def _get_observation_upper_bound(self):
    """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
    upper_bound = np.zeros(self._get_observation_dimension())
    num_motors = self.minitaur.num_motors
    upper_bound[0:num_motors] = math.pi  # Joint angle.
    upper_bound[num_motors:2 * num_motors] = (motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
    upper_bound[2 * num_motors:3 * num_motors] = (motor.OBSERVED_TORQUE_LIMIT)  # Joint torque.
    upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
    return upper_bound

  def _get_observation_lower_bound(self):
    """Get the lower bound of the observation."""
    return -self._get_observation_upper_bound()

  def _get_observation_dimension(self):
    """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
    return len(self._get_observation())

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step

  def set_time_step(self, control_step, simulation_step=0.001):
    """Sets the time step of the environment.

    Args:
      control_step: The time period (in seconds) between two adjacent control
        actions are applied.
      simulation_step: The simulation time step in PyBullet. By default, the
        simulation step is 0.001s, which is a good trade-off between simulation
        speed and accuracy.
    Raises:
      ValueError: If the control step is smaller than the simulation step.
    """
    if control_step < simulation_step:
      raise ValueError("Control step should be larger than or equal to simulation step.")
    self.control_time_step = control_step
    self._time_step = simulation_step
    self._action_repeat = int(round(control_step / simulation_step))
    self._num_bullet_solver_iterations = (NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
    self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=self._num_bullet_solver_iterations)
    self._pybullet_client.setTimeStep(self._time_step)
    self.minitaur.SetTimeSteps(action_repeat=self._action_repeat, simulation_step=self._time_step)

  def get_timesteps(self):
    return self.control_time_step, self._time_step

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def ground_id(self):
    return self._ground_id

  @ground_id.setter
  def ground_id(self, new_ground_id):
    self._ground_id = new_ground_id

  @property
  def env_step_counter(self):
    return self._env_step_counter

  def add_obstacles(self, orientation):
    colBoxId = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,  halfExtents=[0.01, 2, 0.1])
    self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[0, 0, 1.], baseOrientation=[ 0, 0.6427876, 0, 0.7660444 ])

    # visBoxId = self._pybullet_client.createVisualShape(self._pybullet_client.GEOM_BOX, halfExtents=[(obstacle[1]-obstacle[0])/2.0+0.005, (obstacle[3]-obstacle[2])/2.0+0.005, 0.4], rgbaColor=[0.1,0.7,0.6, 1.0], specularColor=[0.3, 0.5, 0.5, 1.0])
    # self._pybullet_client.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0], baseVisualShapeIndex=visBoxId, useMaximalCoordinates=True, basePosition=[(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.04])

  def controller_sawtooth(self, params, t):
        """Sawtooth controller"""
        steer = params[0] if self._unblocked_steering else 0  # Move in different directions
        step_size = params[1]  # Walk with different step_size forward or backward
        leg_extension = params[2]  # Walk on different terrain
        leg_extension_offset = params[3]
        swing_offset = params[4]  # Walk in slopes

        # Robot specific parameters
        swing_limit = 0.6
        extension_limit = 1
        speed = 2.0  # cycle per second
        swing_offset = swing_offset * 0.2

        extension = extension_limit * (leg_extension + 1) * 0.5
        leg_extension_offset = 0.1 * leg_extension_offset - 0.8
        # Steer modulates only the magnitude (abs) of step_size
        A = np.clip(abs(step_size) + steer, 0, 1) if step_size >= 0 else -np.clip(abs(step_size) + steer, 0, 1)
        B = np.clip(abs(step_size) - steer, 0, 1) if step_size >= 0 else -np.clip(abs(step_size) - steer, 0, 1)

        # We want legs to move sinusoidally, smoothly
        fl = math.sin(t * speed * 2 * np.pi) * (swing_limit * A) + swing_offset
        br = math.sin(t * speed * 2 * np.pi) * (swing_limit * B) + swing_offset
        fr = math.sin(t * speed * 2 * np.pi + math.pi) * (swing_limit * B) + swing_offset
        bl = math.sin(t * speed * 2 * np.pi + math.pi) * (swing_limit * A) + swing_offset

        # Sawtooth for faster contraction
        e1 = extension * sawtooth(t, speed, 0.25) + leg_extension_offset
        e2 = extension * sawtooth(t, speed, 0.5 + 0.25) + leg_extension_offset
        return np.clip(np.array([fl, bl, fr, br, -e1, -e2, -e2, -e1]), -1, 1)


def controller_old(t, w, params):
    """general sinusoidal controller"""
    a = [params[0] * np.sin(w * t + params[8]*2*np.pi),   #s FL
         params[1] * np.sin(w * t + params[9]*2*np.pi),   #s BL
         params[2] * np.sin(w * t + params[10]*2*np.pi),  #s FR
         params[3] * np.sin(w * t + params[11]*2*np.pi),  #s BR
         params[4] * np.sin(w * t + params[12]*2*np.pi),  #e FL
         params[5] * np.sin(w * t + params[13]*2*np.pi),  #e BL
         params[6] * np.sin(w * t + params[14]*2*np.pi),  #e FR
         params[7] * np.sin(w * t + params[15]*2*np.pi)   #e BR
         ]
    return a

    # speed
# params = np.array([0.38988197, 0.78974537, 0.42910077, 0.59528132, 0.39941388, 0.45330775,
#     0.50086598, 0.226338,   0.80591036, 0.37183742, 0.55969599, 0.04732664,
#     0.07999865, 0.5300707,  0.46907241, 0.04910368])
    # more speed
# params = np.array([0.5791837,  0.90610882, 0.88157485, 0.7385333,  0.34541331, 0.35552825,
#  0.63303087, 0.3318575,  0.69241517, 0.35977223, 0.53703204, 0.00214748,
#  0.04586218, 0.54583044, 0.47446893, 0.04734582])
    #slow
# params = [0.3]*4+[0.35]*4+ [1.08707590e-04, 7.50718651e-01, 4.76891710e-01, 2.25329594e-01,
# 6.81936235e-02, 7.76331907e-01, 7.79122450e-01, 3.14621179e-01]
# params = [0.3]*4+[0.35]*4 + [0, 0.75, 0.5, 0.25] + [0.0, 0.75, 0.75, 0.25]
# params = [0.3]*4+[0.35]*4 + [0.5, 0.25, 0, 0.75] + [0.75, 0.25, 0., 0.75]


def frac(x):
    return x - np.floor(x)


def sawtooth(t, freq, phase=0):
    T = 1/float(freq)
    y = frac(t/T + phase)
    return y


if __name__ == "__main__":
    import gym
    import time
    import pickle
    import fast_adaptation_embedding.env
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    from pybullet_envs.bullet import minitaur_gym_env
    render = False
    # render = True
    ctrl_time_step = 0.02

    env = gym.make("MinitaurControlledEnv_fastAdapt-v0", render=render, on_rack=0,
                      control_time_step=ctrl_time_step,
                      action_repeat=int(250*ctrl_time_step),
                      accurate_motor_model_enabled=0,
                      pd_control_enabled=1,
                      env_randomizer=None)

    # orientation = [ 0, -0.3826834, 0, 0.9238795 ]
    recorder = None
    Params, R = [], []
    recorder = VideoRecorder(env, "test.mp4")
    Obs, a_action, m_action, position = [], [], [], []
    I = []
    env.set_mismatch([0., 0])
    previous_obs = env.reset()
    env.steering(False)
    # env.add_obstacles(orientation)
    for i in range(50):
        if recorder is not None:
            recorder.capture_frame()
        a = [1, 1, 1, 0, 0]
        obs, r, done, info = env.step(a)
        previous_obs = np.copy(obs)
        # rew += r
        Obs.append(obs)
        a_action.append(info['action'])
        # print(a_action)
        # print(obs)
        # time.sleep(0.02)
        if done:
            break
    with open("foot_pos.pk", 'wb') as f:
        pickle.dump([Obs, a_action, m_action, position], f)

    print(np.sum(info['rewards'], axis=0))
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()





