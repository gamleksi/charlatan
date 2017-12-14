import os
import time
import pybullet as bullet
import numpy as np
from pybullet_envs.bullet import KukaGymEnv
from pybullet_envs.bullet import kuka
from gym import spaces
import pybullet_data


class KukaTorqueControl(kuka.Kuka):
    def __init__(self, urdfRootPath, timeStep):
        super(KukaTorqueControl, self).__init__(urdfRootPath=urdfRootPath, timeStep=timeStep)

    def reset(self):
        # not necessary (?)
        super(KukaTorqueControl, self).reset()
        for jointIndex in range (self.numJoints):
            bullet.resetJointState(self.kukaUid,jointIndex,self.jointPositions[jointIndex])
            bullet.setJointMotorControl2(bodyUniqueId=self.kukaUid,   jointIndex=jointIndex, controlMode=bullet.TORQUE_CONTROL)

    def applyAction(self, motorCommands):
        for idx in range(len(motorCommands)):
            motor = self.motorIndices[idx]
            bullet.setJointMotorControl2(self.kukaUid, motor,bullet.TORQUE_CONTROL, force=motorCommands[idx])

class KukaPositionControl(kuka.Kuka):
    def __init__(self, urdfRootPath, timeStep):
        super(KukaPositionControl, self).__init__(urdfRootPath=urdfRootPath, timeStep=timeStep)

    def reset(self):
        super(KukaPositionControl, self).reset()
        self.numJoints = bullet.getNumJoints(self.kukaUid)
        for jointIndex in range (self.numJoints):
            joint_info = bullet.getJointInfo(self.kukaUid, jointIndex)
            min_pos = joint_info[8]
            max_pos = joint_info[9]
            joint_pos = np.random.uniform(low=min_pos, high=max_pos)
            bullet.resetJointState(self.kukaUid,jointIndex, joint_pos)
            bullet.setJointMotorControl2(self.kukaUid, jointIndex,bullet.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=self.maxForce
                )

class KukaPoseEnv(KukaGymEnv):
    # The goal in this env is to get the robot in a given pose.
    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=1,
            isEnableSelfCollision=True, renders=True, goalReset=True, goal=None):
        self._timeStep = 1./240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = None
        self._envStepCounter = 0
        self._renders = renders
        self.terminated = 0
        self._setup_rendering()
        self._setup_kuka()
        self._setup_spaces()
        self._goalReset = goalReset
        self.goal = np.array(goal) if goal is not None else self.getNewGoal()
        self.joint_history = []
        self._seed()
        #self.reset()
        self.viewer = None

    def _setup_rendering(self):
        if self._renders:
            cid = bullet.connect(bullet.SHARED_MEMORY)
            if (cid<0):
                cid = bullet.connect(bullet.GUI)
            bullet.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
        else:
            bullet.connect(bullet.DIRECT)

    def _setup_kuka(self):
        self._kuka = KukaTorqueControl(self._urdfRoot, self._timeStep)
        self._kuka.useInverseKinematics = False

    def _setup_action_space(self):
        action_dimensions = len(self._kuka.motorIndices)
        self.action_space = spaces.Box(
            low=-np.ones(action_dimensions) * self._kuka.maxForce,
            high=np.ones(action_dimensions) * self._kuka.maxForce)

    def _setup_observation_space(self):
         self.observation_space = spaces.Box(
            low=np.concatenate((self.joint_lower_limit,-self.joint_velocity_limit, self.joint_lower_limit)),
            high=np.concatenate((self.joint_upper_limit, self.joint_velocity_limit, self.joint_upper_limit)))

    def _setup_goal_space(self):
        self.goal_space = spaces.Box(
            low=self.joint_lower_limit,
            high=self.joint_upper_limit)

    def _setup_spaces(self):
        motor_count = len(self._kuka.motorIndices)
        self.joint_lower_limit = np.zeros(motor_count)
        self.joint_upper_limit = np.zeros(motor_count)
        self.joint_velocity_limit = np.zeros(motor_count)
        for i, joint_index in enumerate(self._kuka.motorIndices):
            joint_info = bullet.getJointInfo(self._kuka.kukaUid, joint_index)
            if joint_info != bullet.JOINT_FIXED:
                self.joint_lower_limit[i] = joint_info[8]
                self.joint_upper_limit[i] = joint_info[9]
                self.joint_velocity_limit[i] = joint_info[11]
        self._setup_action_space()
        self._setup_observation_space()
        self._setup_goal_space()


    def _reset(self):
        bullet.resetSimulation()
        bullet.setPhysicsEngineParameter(numSolverIterations=150, fixedTimeStep=1/30)
        bullet.setTimeStep(self._timeStep)
        bullet.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"),[0, 0, 0])
        bullet.setGravity(0, 0, -10)

        self._kuka.reset()
        self._envStepCounter = 0
        bullet.stepSimulation()

        self.terminated = 0
        # Sample a new random goal pose
        if self._goalReset:
            self.goal = self.getNewGoal()
        self._observation = self.getExtendedObservation()

        return self._observation

    def _termination(self):
        too_long = self._envStepCounter > 1e4
        at_goal = np.linalg.norm(self._motorized_joint_positions() - self.goal, 2) < 1e-3
        return too_long or at_goal

    def _step(self, action):
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            bullet.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self.getExtendedObservation()
            self._envStepCounter += 1
            done = self._termination()
            if done:
                break
        # self.last_action = action
        reward = self._reward()
        return self._observation, reward, done, {}

    def _reward(self):
        goal_distance = -np.linalg.norm(np.abs(self._motorized_joint_positions() - self.goal), 2)
        return goal_distance # - np.linalg.norm(np.abs(self.last_action), 2)

    def getNewGoal(self):
        goal = self.goal_space.sample()
        return np.array(goal)

    def _motorized_joint_positions(self):
        joint_states = bullet.getJointStates(self._kuka.kukaUid, self._kuka.motorIndices)
        return np.array([jointState[0] for jointState in joint_states])

    def _joint_velocities(self):
        joint_states = bullet.getJointStates(self._kuka.kukaUid, self._kuka.motorIndices)
        return np.array([jointState[1] for jointState in joint_states])

    def getExtendedObservation(self):
        # The first item in the joint state is the joint position.
        joint_positions = self._motorized_joint_positions()
        joint_velocities = self._joint_velocities()

        return np.concatenate((joint_positions, joint_velocities, self.goal))

from util import _resize_frame

class KukaTrajectoryEnv(KukaPoseEnv):
    CAMERA_POS = {
        'distance': 2.3,
        'pitch': -20,
        'yaw': 0,
    }
    CAMERA_OFFSET = np.array([0, 0, 0.3])

    def __init__(self, record_movement=True, **kwargs):
        self.record_movement = record_movement
        super(KukaTrajectoryEnv, self).__init__(**kwargs)

    def _setup_action_space(self):
        self.action_space = spaces.Box(
            low=self.joint_lower_limit,
            high=self.joint_upper_limit)

    def _setup_observation_space(self):
        self.observation_space = spaces.Box(
            low=self.joint_lower_limit,
            high=self.joint_upper_limit)

    def _setup_kuka(self):
        self._kuka = KukaPositionControl(self._urdfRoot, self._timeStep)
        self._kuka.maxForce = 500
        self._kuka.useInverseKinematics = False

    def _get_current_frame(self):
        base_position, _ = bullet.getBasePositionAndOrientation(self._kuka.kukaUid)
        view_matrix = bullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_position + self.CAMERA_OFFSET,
            distance=self.CAMERA_POS['distance'],
            yaw=self.CAMERA_POS['yaw'],
            pitch=self.CAMERA_POS['pitch'],
            roll=0,
            upAxisIndex=2)
        proj_matrix = bullet.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        _, _, px, _, _ = bullet.getCameraImage(
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            width=304, height=304, renderer=bullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px[:, :, 0:3])
        return rgb_array

    def _step(self, action):
        observation, reward, done, _ = super(KukaTrajectoryEnv, self)._step(action)
        if self.record_movement:
            frame = self._get_current_frame()
            return observation, reward, done, frame
        else:
            return observation, reward, done, _

    def getExtendedObservation(self):
        joint_positions = self._motorized_joint_positions()
        return joint_positions