# Insert packages path to sys.path
# So that the codes can be run from any directory

import sys
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

this_file_path = os.path.dirname(os.path.realpath(__file__))
import gym, gym.spaces, gym.utils, gym.utils.seeding
import pybullet
import pybullet_data
import time
import numpy as np
import math
import copy

base_path = this_file_path


class PexodQuad_env(gym.Env):
    "Hexapod environment"
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, execution_time=1.0, simStep=0.004, controlStep=0.02, controller=None,
                 jointControlMode="position", visualizationSpeed=1.0,
                 boturdf=base_path + "/assets/urdf/pexod_quad/pexod_quad.urdf",
                 floorurdf=base_path + "/assets/urdf/pexod_quad/plane.urdf", lateral_friction=10.0,
                 xreward=1, yreward=0):
        super(PexodQuad_env, self).__init__()
        self.action_space = gym.spaces.Box(low=-np.ones(4), high=np.ones(4), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.p = pybullet
        self.__vspeed = visualizationSpeed
        self.__init_state = [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 0.0]  # position and orientation
        self.__simStep = simStep
        self.__controlStep = controlStep
        self.__controller = HexaControllerSine()  # controller
        self.physicsClient = object()
        assert jointControlMode in ["position", "velocity"]
        self.jointControlMode = jointControlMode  # either "position" or "velocity"
        self.__base_collision = False
        self.__heightExceed = False
        self.boturdf = boturdf
        self.floorurdf = floorurdf
        self.lateral_friction = lateral_friction
        self.__env_created = False
        self.render_mode = 'rgb_array'
        self.execution_time = execution_time  # seconds
        self.recorder = None
        # Mismatches : 0-11 : multiplies joint actions, 12: Adds 360.0*self.__mismatches[12] to z rotation angle
        self.__mismatches = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.])

        self._cam_dist = 2
        self._cam_yaw = 0
        self._cam_pitch = -50
        self._render_width = 500
        self._render_height = int(self._render_width * 720 / 1280)
        self._current_time = 0
        self._xreward = xreward
        self._yreward = yreward

    def setFriction(self, lateral_friction):
        self.p.changeDynamics(self.__planeId, linkIndex=-1, lateralFriction=lateral_friction,
                              physicsClientId=self.physicsClient)

    def get_simulator(self):
        return self.p, self.physicsClient

    def _make_joint_list(self, botId):
        joint_names = [b'body_leg_0', b'leg_0_1_2', b'leg_0_2_3',
                       b'body_leg_2', b'leg_2_1_2', b'leg_2_2_3',
                       b'body_leg_3', b'leg_3_1_2', b'leg_3_2_3',
                       b'body_leg_5', b'leg_5_1_2', b'leg_5_2_3',
                       ]
        joint_list = []
        for n in joint_names:
            for joint in range(self.p.getNumJoints(botId, physicsClientId=self.physicsClient)):
                name = self.p.getJointInfo(botId, joint, physicsClientId=self.physicsClient)[1]
                if name == n:
                    joint_list += [joint]
        return joint_list

    def setRecorder(self, recorder):
        self.recorder = recorder

    def step(self, action):
        self.__controller.setParams(action)

        self.__base_collision = False
        self.__heightExceed = False

        assert not self.__controller == None, "Controller not set"
        self.__states = [self._getState()]  # load with init state
        self.__rotations = [self.getRotation()]
        self.__commands = []
        old_time = 0
        first = True
        self.__command = object

        self.flip = False
        for i in range(int(self.__controlStep / self.__simStep)):
            if first:
                self.__command = self.__controller.nextCommand(self._current_time) * self.__mismatches[0:12]
                self.__commands.append(self.__command)
                first = False
                if self.recorder is not None:
                    self.recorder.capture_frame()

            self.set_commands(self.__command)
            self.p.stepSimulation(physicsClientId=self.physicsClient)
            self._current_time += self.__simStep
            if self.p.getConnectionInfo(self.physicsClient)['connectionMethod'] == self.p.GUI:
                time.sleep(self.__simStep / float(self.__vspeed))

                # Flipfing behavior
            if self.getZOrientation() < 0.0:
                self.flip = True

            # Base floor collision
            if len(self.p.getContactPoints(self.__planeId, self.__hexapodId, -1, -1,
                                           physicsClientId=self.physicsClient)) > 0:
                self.__base_collision = True

            # Jumping behavior when CM crosses 2.2
            if self._getState()[2] > 2.2:
                self.__heightExceed = True

        self.__states.append(self._getState())
        self.__rotations.append(self.getRotation())
        state = self.state
        info = {'acs': self.__command[[0, 1, 4, 6]], 'obs': list(state) + self.get_true_observation()}
        rew = (state[0] - self.previous_state[0]) * self._xreward + (state[1] - self.previous_state[1]) * self._yreward
        self.previous_state = np.copy(state)
        return state, rew, copy.copy(self.flip), info

    def isBaseCollision(self):
        return self.__base_collision

    def isHeightExceed(self):
        return self.__heightExceed

    def states(self):
        return self.__states

    def rotations(self):
        return self.__rotations

    def commands(self):
        return self.__commands

    def simStep(self):
        return self.__simStep

    def controlStep(self):
        return self.__controlStep

    def get_true_observation(self):
        angles, velocities, torques = [], [], []
        for joint in self.joint_list:
            info = self.p.getJointState(self.__hexapodId, joint, physicsClientId=self.physicsClient)
            angles.append(info[0])
            velocities.append(info[1])
            torques.append(info[3])
        return angles + velocities #+ torques

    # command should be numpy array
    def set_commands(self, commands):
        assert commands.size == len(self.joint_list), "Command length doesn't match with controllable joints"
        counter = 0
        for joint in self.joint_list:
            info = self.p.getJointInfo(self.__hexapodId, joint, physicsClientId=self.physicsClient)
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            pos = min(max(lower_limit, commands[counter]), upper_limit)

            if self.jointControlMode == "position":
                self.p.setJointMotorControl2(bodyUniqueId=self.__hexapodId, jointIndex=joint,
                                             controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=commands[counter],
                                             force=max_force,
                                             maxVelocity=max_velocity,
                                             physicsClientId=self.physicsClient)

            elif self.jointControlMode == "velocity":
                current_joint_pos = self.p.getJointState(bodyUniqueId=self.__hexapodId, jointIndex=joint,
                                                         physicsClientId=self.physicsClient)[0]
                err = pos - current_joint_pos
                self.p.setJointMotorControl2(bodyUniqueId=self.__hexapodId, jointIndex=joint,
                                             controlMode=self.p.VELOCITY_CONTROL,
                                             # velocity must be limited as it it not done automatically
                                             # max_force is however considered here
                                             targetVelocity=np.clip(err * (1.0 / (math.pi * self.__controlStep)),
                                                                    -max_velocity, max_velocity),
                                             force=max_force,
                                             physicsClientId=self.physicsClient)
            counter = counter + 1

    @property
    def state(self):
        pos = self._getState()[0:2]
        z_ang_sin = np.sin(np.deg2rad(self.getEulerAngles()[2] + 360.0 * self.__mismatches[12]))
        z_ang_cos = np.cos(np.deg2rad(self.getEulerAngles()[2] + 360.0 * self.__mismatches[12]))
        self._state = np.array([pos[0], pos[1], z_ang_sin, z_ang_cos])
        return self._state

    def _getState(self):
        '''
        Returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order.
        Use self.p.getEulerFromQuaternion to convert the quaternion to Euler if needed.
        '''
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        pos = list(states[0])
        orient = list(states[1])
        return pos + orient

    def getZOrientation(self):
        '''
        Returns z component of up vector of the robot
        It is negative if the robot is flipped.
        '''
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        z_componentOfUp = self.p.getMatrixFromQuaternion(list(states[1]))[-1]
        return z_componentOfUp

    def getRotation(self):
        '''
        Returns rotation matrix
        '''
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        return self.p.getMatrixFromQuaternion(list(states[1]))

    def getEulerAngles(self):
        states = self.p.getBasePositionAndOrientation(self.__hexapodId, physicsClientId=self.physicsClient)
        euler_angles = self.p.getEulerFromQuaternion(list(states[1]))  # x,y,z rotation
        return np.rad2deg(euler_angles)

    def flipped(self):
        return self.flip

    def render(self, mode='rgb_array', close=False):
        self.render_mode = mode
        if self.render_mode == 'human' or not self.__env_created:
            return np.array([])

        base_pos = self._getState()[0:3]

        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def joint_reset(self):
        self.step(np.array([0., 0., -1.0, 0.0]))
        return self.state

    def reset(self):
        '''
        orientation must be in quarternion
        '''
        if not self.__env_created:
            if self.render_mode == 'human':
                self.physicsClient = self.p.connect(self.p.GUI)
            else:
                self.physicsClient = self.p.connect(self.p.DIRECT)

            self.p.resetSimulation(physicsClientId=self.physicsClient)
            self.p.setTimeStep(self.__simStep, physicsClientId=self.physicsClient)
            self.p.resetDebugVisualizerCamera(1.5, 50, -35.0, [0, 0, 0], physicsClientId=self.physicsClient)
            self.p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physicsClient)
            self.p.setGravity(0, 0, -10.0, physicsClientId=self.physicsClient)

            self.__planeId = self.p.loadURDF(self.floorurdf, [0, 0, 0], self.p.getQuaternionFromEuler([0, 0, 0]),
                                             physicsClientId=self.physicsClient)
            self.hexapodStartPos = [0, 0, 0.2]  # Start at collision free state. Otherwise results becomes a bit random
            self.hexapodStartOrientation = self.p.getQuaternionFromEuler([0, 0, 0])
            flags = self.p.URDF_USE_INERTIA_FROM_FILE or self.p.URDF_USE_SELF_COLLISION or self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
            self.__hexapodId = self.p.loadURDF(self.boturdf, self.hexapodStartPos, self.hexapodStartOrientation,
                                               useFixedBase=0, flags=flags, physicsClientId=self.physicsClient)

            self.joint_list = self._make_joint_list(self.__hexapodId)
            self.goal_position = []
            self.goalBodyID = None
            self.visualGoalShapeId = self.p.createVisualShape(shapeType=self.p.GEOM_CYLINDER, radius=0.2, length=0.04,
                                                              visualFramePosition=[0., 0., 0.],
                                                              visualFrameOrientation=self.p.getQuaternionFromEuler(
                                                                  [0, 0, 0]), rgbaColor=[0.0, 0.0, 0.0, 0.5],
                                                              specularColor=[0.5, 0.5, 0.5, 1.0],
                                                              physicsClientId=self.physicsClient)

            self.p.changeDynamics(self.__planeId, linkIndex=-1, lateralFriction=self.lateral_friction,
                                  physicsClientId=self.physicsClient)
            self.__env_created = True

        for i in range(self.p.getNumJoints(self.__hexapodId, physicsClientId=self.physicsClient)):
            self.p.resetJointState(self.__hexapodId, i, 0.0, 0.0, physicsClientId=self.physicsClient)

        pos = self.hexapodStartPos
        orient = self.hexapodStartOrientation
        self.p.resetBasePositionAndOrientation(self.__hexapodId, pos, orient, physicsClientId=self.physicsClient)

        self.__states = []
        self.__rotations = []
        self.__commands = []
        pos = self._getState()[0:2]
        self._current_time = 0.
        self.previous_state = np.copy(self.state)
        return list(self.state) #+ self.get_true_observation()

    def get_current_time(self):
        return self._current_time

    def disconnet(self):
        self.p.disconnect(physicsClientId=self.physicsClient)

    def setController(self, controller):
        self.__controller = controller

    def getController(self):
        return self.__controller

    def set_mismatch(self, mismatches):
        self.__mismatches = mismatches


class GenericController:
    "A generic controller. It need to be inherited and overload for specific controllers"

    def __init__(self):
        pass

    def nextCommand(self, CurrenState=np.array([0]), timeStep=0):
        raise NotImplementedError()

    def setParams(self, params=np.array([0])):
        raise NotImplementedError()

    def setRandom(self):
        raise NotImplementedError()

    def getParams(self):
        return self._params

    def setCommandLimits(sef, limits):
        pass


class HexaControllerSine(GenericController):

    def __init__(self, params=None, array_dim=100):
        self.array_dim = array_dim
        self._params = None

    def nextCommand(self, t):
        # Control parameters
        # steer = 0.0 #Move in different directions
        # step_size = 1.0 # Walk with different step_size forward or backward
        # leg_extension = 1.0 #Walk on different terrain
        # leg_extension_offset = -1.0

        steer = self._params[0]  # Move in different directions
        step_size = self._params[1]  # Walk with different step_size forward or backward
        leg_extension = self._params[2]  # Walk on different terrain
        leg_extension_offset = self._params[3]

        # Robot specific parameters
        swing_limit = 0.5
        extension_limit = 0.4
        speed = 2

        A = np.clip(step_size + steer, -1, 1)
        B = np.clip(step_size - steer, -1, 1)
        extension = extension_limit * (leg_extension + 1.0) * 0.5
        max_extension = np.clip(extension + extension_limit * leg_extension_offset, 0, extension)
        min_extension = np.clip(-extension + extension_limit * leg_extension_offset, -extension, 0)

        # We want legs to move sinusoidally, smoothly
        fl = math.sin(t * speed * 2 * np.pi) * (swing_limit * A)
        br = math.sin(t * speed * 2 * np.pi) * (swing_limit * B)
        fr = math.sin(t * speed * 2 * np.pi + math.pi) * (swing_limit * B)
        bl = math.sin(t * speed * 2 * np.pi + math.pi) * (swing_limit * A)

        # We can legs to reach extreme extension as quickly as possible: More like a smoothed square wave
        e1 = np.clip(3.0 * math.sin(t * speed * 2 * np.pi + math.pi / 2), min_extension, max_extension)
        e2 = np.clip(3.0 * math.sin(t * speed * 2 * np.pi + math.pi + math.pi / 2), min_extension, max_extension)
        # return np.array([bl,e1,e1, 0,0,0, fl,e2,e2, -fr,e1,e1, 0,0,0, -br,e2,e2]) * np.pi/4
        return np.array([bl, e1, e1, fl, e2, e2, -fr, e1, e1, -br, e2, e2]) * np.pi / 4

        # Swing
        # 0: back_left
        # 6: front_left
        # 9: front_right
        # 15: back_right

    def setParams(self, params, array_dim=100):
        self._params = params

    def setRandom(self):
        self._random = True
        self.setParams(np.random.rand(4) * 2.0 - 1.0)

    def getParams(self):
        return self._params


if __name__ == '__main__':
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    import time
    import fast_adaptation_embedding.env
    import pickle
    env = gym.make("PexodQuad-v0", visualizationSpeed=2, simStep=0.004, controlStep=0.02, jointControlMode="position")
    recorder = None
    from tqdm import tqdm
    # recorder = VideoRecorder(env, "temp_test.mp4")
    env.render(mode='human')
    env.setRecorder(recorder)
    env.reset()
    # States, R, Controller = [], [], []
    pp_new = env.action_space.sample()
    env.setRecorder = recorder
    r = 0
    for i in range(400):
        state, rew, flip, info = env.step(action=pp_new)
        r += rew
    # States.append(np.copy(state))
    # R.append(r)
    # Controller.append(np.copy(pp_new))

    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    # with open('data/pexod.pk', 'wb') as f:
    #     pickle.dump([States, R, Controller], f)
