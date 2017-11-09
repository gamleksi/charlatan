import matplotlib.pyplot as plt
from numpy import genfromtxt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-files', type=int, default=1,
    help='how many files from tmp/gym file')
    parser.add_argument('--dir', default='../pytorch-a2c-ppo-acktr/tmp/gym/')
    parser.add_argument('--source-name', default='.monitor.csv')
    parser.add_argument('--num-mean', type=int, default=500)
    parser.add_argument('--save-name', default='learning')
    return parser.parse_args()

args = get_args()

fig, axes = plt.subplots(args.num_files, 1, figsize=(12, 40))

for index in range(0, args.num_files):
    monitor = genfromtxt("{}{}{}".format(args.dir, index, args.source_name), delimiter=',')
    monitor =  monitor[1:,0]
    iterations = monitor.shape[0]
    i = 0
    arr = []
    while (i + 1) * args.num_mean < iterations:
        arr.append(monitor[i * args.num_mean: (i+1) * args.num_mean].mean())
        i += 1

    if(args.num_files == 1):
        ax = axes
    else:
        ax = axes[index]
    ax.plot(range(1, len(arr) + 1), arr)
    ax.title.set_text("Monitor {}".format(index))

print("Number of updates: {}".format(monitor.shape[0]))

plt.ylabel('Average reward per {} updates'.format(args.num_mean))
plt.savefig("{}.png".format(args.save_name))
