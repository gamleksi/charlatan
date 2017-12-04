import os, sys
import gym
from kuka.env import KukaPoseEnv
import csv

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytorch_a2c_ppo_acktr'))
from pytorch_a2c_ppo_acktr.main import main as ppo
from pytorch_a2c_ppo_acktr.arguments import get_args

from baselines import bench

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id) 
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

def save_parameters(args):

    parameters = vars(args)
    values = list(parameters.values())
    keys = list(parameters.keys())
    
    if  not(os.path.isfile('experiments.csv')):
        with open('experiments.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open('experiments.csv', 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)

def main():

    args = get_args() # list of all arguments: pytorch_a2c_ppo_acktr/arguments.py

    number_of_processor = args.num_processes
    env_id = 'KukaImitatorEnv-v0'
    log_dir = args.log_dir
    envs = []
    for i in range(0, number_of_processor):
        envs.append(make_env(env_id, 2222, i, log_dir))
    

    # save_parameters(args)
    ppo(envs=envs)

if __name__ == "__main__":
    main()