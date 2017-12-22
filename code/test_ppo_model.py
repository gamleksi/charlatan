import os, sys
import gym
from kuka.env import KukaPoseEnv
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytorch_a2c_ppo_acktr'))
from pytorch_a2c_ppo_acktr.enjoy import main as enjoy 
from pytorch_a2c_ppo_acktr.arguments import get_args


def main():

    args = get_args() # list of all arguments: pytorch_a2c_ppo_acktr/arguments.py
    env = gym.make(args.env_name) 
    env.seed(2222 + 2)
    actor_critic = torch.load(os.path.join('trained_models/ppo/', args.model_name+ ".pt"))
    enjoy(env=env, actor_critic=actor_critic)

if __name__ == "__main__":
    main()