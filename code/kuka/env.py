import os
import time
import pybullet as bullet
import numpy as np
from pybullet_envs.bullet import KukaGymEnv
from pybullet_envs.bullet import kuka
from gym import spaces

class KukaPoseEnv(KukaGymEnv):
    # The goal of this env is to get the robot in a given pose.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._kuka.useInverseKinematics = False

        self.action_space = spaces.Box(
            low=self._kuka.ll,
            high=self._kuka.ul)
        self.observation_space = spaces.Box(
            low=np.zeros_like(self._kuka.jr),
            high=self._kuka.jr)

    def _reset(self):
        self.terminated = 0
        bullet.resetSimulation()
        bullet.setPhysicsEngineParameter(numSolverIteration=150)
        bullet.setTimeStep(self._timeStep)
        bullet.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"),[0, 0, -1])
        bullet.setGravity(0, 0, -10)

        self._kuka.reset()
        self._envStepCounter = 0
        bullet.stepSimulation()
        self._observation = self.getExtendedObservation()

        # Sample a new random goal pose
        self.goal = self.observation_space.sample()
        return np.array(self._observation)

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
        return self._kuka.getObservation()

