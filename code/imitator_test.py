import os, sys
import gym
from kuka.env import KukaPoseEnv
import csv

import torch
import pybullet as bullet
import numpy as np
from torch.autograd import Variable
import argparse

def get_args():

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytorch_a2c_ppo_acktr'))


    parser = argparse.ArgumentParser(description='Imitator test')

    parser.add_argument('--env-name', default="KukaImitatorEnv-v0",
                        )
    parser.add_argument('--model-name', default='KukaPoseEnv-v0',
                        help='environment to train on (default: KukaPoseEnv-v0)')
    parser.add_argument('--model-dir', default='./pytorch_a2c_ppo_acktr/trained_models/ppo/', help='load model from directory')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4) Needs to be the same when the model was trained')
    return parser.parse_args()

def main():

    args = get_args()
    env = gym.make(args.env_name)

    # actor_critic = torch.load(os.path.join(args.model_dir, args.model_name))
    # actor_critic.eval()

    # obs_shape = (env.observation_space.shape[0] * args.num_stack,)
    # current_state = torch.zeros(1, *obs_shape)
    
    action_space = env.action_space.shape[0]
    def update_current_state(state):
        shape_dim0 = env.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    # building the environment
    env.render('human')
    state = env.reset()
    # update_current_state(state)

    steps = 0
    done = False

    total_reward = 0
    while not(done):

        # value, action = actor_critic.act(Variable(current_state, volatile=True),deterministic=True)

        # action = action.data.cpu().numpy()[0]
        action = np.zeros(action_space) + 0.2
        state, reward, done, _ = env.step(action)
        total_reward += reward
        # update_current_state(state)
        steps += 1

if __name__ == "__main__":
    main()