import gym
from kuka.env import KukaPoseEnv
import numpy as np

env = gym.make("KukaPoseEnv-v0")


for i in range(10):
    done = False
    env.reset()
    action = env.action_space.sample()
    while not done:
        observation, reward, done, _ = env.step(action)
        env.render()