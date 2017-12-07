import os, sys
import gym
import csv

from imitation import ImitationEnv
import torch


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytorch_a2c_ppo_acktr'))
from pytorch_a2c_ppo_acktr.main import main as ppo
from pytorch_a2c_ppo_acktr.arguments import get_args
from baselines import bench
from util import normalize

def make_env(tcn, frame_size, transforms, seed, rank, log_dir):
    def _thunk():
        env = ImitationEnv(video_dir='./data/video/angle-1', tcn=tcn, frame_size=frame_size, transforms=None, renders=False)
        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

def save_parameters(args):

    parameters = vars(args)
    values = list(parameters.values())
    keys = list(parameters.keys())
    
    if  not(os.path.isfile('imitator_experiments.csv')):
        with open('imitator_experiments.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open('imitator_experiments.csv', 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)

from tcn import define_model, PosNet

def load_model(use_cuda):
    tcn = define_model(use_cuda)
    model_path = os.path.join(
        "./trained_models/tcn",
        "inception-epoch-2000.pk"
    )
    tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    return tcn


def main():

    args = get_args() # list of all arguments: pytorch_a2c_ppo_acktr/arguments.py

    tcn = load_model(args.cuda)
    transforms = normalize
    frame_size = (299, 299)

    number_of_processor = args.num_processes
    log_dir = args.log_dir
    envs = []
    for i in range(0, number_of_processor):
        envs.append(make_env(tcn, frame_size, transforms, 2222, i, log_dir))
    save_parameters(args)
    ppo(envs=envs)

if __name__ == "__main__":
    main()