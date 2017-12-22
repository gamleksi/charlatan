import argparse
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytorch_a2c_ppo_acktr'))

from pytorch_a2c_ppo_acktr.visualize import load_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='./tmp/gym/')
    parser.add_argument('--source-name', default='.monitor.csv')
    parser.add_argument('--num-mean', type=int, default=5)
    parser.add_argument('--save-name', default='learning')
    parser.add_argument('--title', default='Learning')

    return parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np

def imitation_plot(title, episodes=539):

    file = os.path.join('./data/imitation_log/', '{}.csv'.format(title))
    data = []
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(',')
            tmp = np.array([float(i) for i in tmp])
            data.append(tmp)
    data = np.array(data)
    drop_num = (data.shape[0] // episodes) * episodes
    data = data[:drop_num]
    data = np.array_split(data, data.shape[0]/episodes)
    episode_mean = np.array([np.sum(episodes, axis=0).mean() for episodes in data])
    fig = plt.figure()
    plt.plot( np.arange(1, episode_mean.shape[0] + 1),episode_mean, label="{}".format("Imitation Learning"))   
    plt.xlabel('Updates')
    plt.savefig('{}.png'.format(title))


def main():
    args = get_args()
    tx, ty = load_data(args.dir, 1, 100)
    if tx is None:
        import ipdb; ipdb.set_trace()
        print("Data not found")
        return

    import matplotlib.pyplot as plt
    plt.plot(tx, ty)

    #plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
    #            ["1M", "2M", "4M", "6M", "8M", "10M"])
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(args.title)
    plt.show()
    plt.savefig("{}.png".format(args.save_name))

if __name__ == "__main__":
    #main()
    imitation_plot("KukaImitation-v2")