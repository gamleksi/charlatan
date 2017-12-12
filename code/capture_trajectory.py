import gym
from kuka.env import KukaPoseEnv
import numpy as np
from pybullet_envs.bullet import KukaGymEnv
from util import write_video 
from PIL import Image


def main(path, record_movement):
    env = gym.make('KukaTrajectoryEnv-v1')
    for i in range(10):
        done = False
        env.reset()
        action = env.action_space.sample()
        frames = []
        while not done:
            observation, reward, done, frame = env.step(action)
            frames.append(np.array(frame))
            #img = Image.fromarray(frame)
            #img.show()
        if(record_movement):
            write_video("{}_trajectory.mp4".format(i), path, frames)

if __name__ == '__main__':
    path = "./tmp/video/"
    record_movement = True 
    main(path, record_movement)