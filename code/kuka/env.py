import os
import time
import pybullet as bullet
import numpy as np
from pybullet_envs.bullet import KukaGymEnv
from pybullet_envs.bullet import kuka
from gym import spaces
import pybullet_data

class KukaPoseEnv(KukaGymEnv):
    # The goal in this env is to get the robot in a given pose.
    def __init__(self, urdfRoot=pybullet_data.getDataPath(), actionRepeat=1,
            isEnableSelfCollision=True, renders=True):
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

        self._seed()
        self.reset()
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
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._kuka.useInverseKinematics = False

    def _setup_spaces(self):
        action_dimensions = len(self._kuka.motorIndices)
        self.action_space = spaces.Box(
            low=-np.ones(action_dimensions) * self._kuka.maxForce,
            high=np.ones(action_dimensions) * self._kuka.maxForce)

        joint_lower_limit = np.zeros(self._kuka.numJoints)
        joint_upper_limit = np.zeros(self._kuka.numJoints)
        for joint_index in range(self._kuka.numJoints):
            joint_info = bullet.getJointInfo(self._kuka.kukaUid, joint_index)
            joint_lower_limit[joint_index] = joint_info[8]
            joint_upper_limit[joint_index] = joint_info[9]

        self.observation_space = spaces.Box(
            low=joint_lower_limit,
            high=joint_upper_limit)

    def _reset(self):
        self.terminated = 0
        bullet.resetSimulation()
        bullet.setPhysicsEngineParameter(numSolverIterations=150)
        bullet.setTimeStep(self._timeStep)
        bullet.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"),[0, 0, 0])
        bullet.setGravity(0, 0, -10)

        self._kuka.reset()
        self._envStepCounter = 0
        bullet.stepSimulation()
        self._observation = self.getExtendedObservation()

        # Sample a new random goal pose
        self.goal = self.observation_space.sample()
        return np.array(self._observation)

    def _termination(self):
        too_long = self._envStepCounter > 1e4
        at_goal = np.linalg.norm(self._observation - self.goal, 2) < 1e-6
        return too_long or at_goal

    def _step(self, action):
        self._kuka.applyAction(action)
        for i in range(self._actionRepeat):
            bullet.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self.getExtendedObservation()
            done = self._termination()
            if done:
                break
            self._envStepCounter += 1
        reward = self._reward()
        return np.array(self._observation), reward, done, {}

    def _reward(self):
        return -np.linalg.norm(np.abs(self._observation - self.goal), 2)

    def getExtendedObservation(self):
        joint_states = bullet.getJointStates(self._kuka.kukaUid, range(self._kuka.numJoints))
        # The first item in the joint state is the joint position.
        joint_positions = [jointState[0] for jointState in joint_states]
        return np.array(joint_positions)

