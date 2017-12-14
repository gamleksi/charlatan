import argparse
import gym
from kuka.env import KukaPoseEnv
import numpy as np
from pybullet_envs.bullet import KukaGymEnv
from util import write_video, ensure_folder
from PIL import Image
import multiprocessing


def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='tmp/')
    parser.add_argument('-n', default=10, type=int)
    parser.add_argument('--no-write', action='store_true')
    parser.add_argument('--num-processes', type=int, default=4)
    args = parser.parse_args()
    args.write = not args.no_write
    return args

REWARD_CUTOFF = 1.0

def generate_trajectory(i, env, args):
    done = False
    env.reset()
    action = env.action_space.sample()
    frames = []
    while not done:
        _, reward, done, frame = env.step(env.env.goal)
        frames.append(np.array(frame))
        #img = Image.fromarray(frame)
        #img.show()
    # reward check to make sure the end point is reachable.
    if reward < REWARD_CUTOFF and args.write:
        print("writing: ", i)
        write_video("{}.mp4".format(i), args.out, frames)
    elif reward < REWARD_CUTOFF:
        generate_trajectory(i, env, args)

class TrajectoryMachine(object):
    def __init__(self):
        self.env = gym.make('KukaTrajectoryEnv-v1')

    def __call__(self, queue, args):
        while 1:
            try:
                i = queue.get(timeout=1)
                generate_trajectory(i, self.env, args)
            except (multiprocessing.TimeoutError, multiprocessing.Queue.Empty):
                break

def main(args):
    ensure_folder(args.out)
    queue = multiprocessing.Queue(args.n)
    for i in range(args.n):
        queue.put(i)

    processes = []
    for i in range(args.num_processes):
        process = multiprocessing.Process(target=TrajectoryMachine(), args=(queue, args))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()

if __name__ == '__main__':
    path = "./tmp/video/"
    main(get_cli_args())