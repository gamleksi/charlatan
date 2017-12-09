import gym
from kuka.env import KukaPoseEnv

env = gym.make("KukaPoseEnv-v0")

env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    import ipdb; ipdb.set_trace()
    env.render()