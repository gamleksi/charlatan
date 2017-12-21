import os
import numpy as np
import torch
from torch import optim
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from util import read_video, normalize
from torchvision import transforms
from tcn import define_model
from sklearn import manifold
import matplotlib
from matplotlib import pyplot as plot


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-video', type=str, default='./data/validation/angle1/16.mp4')
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args()

def load_model(model_path):
    tcn = define_model(pretrained=False)
    tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    return tcn

def calculate_distance_to_next(embeddings):
    out = np.zeros(embeddings.shape[0], dtype=np.float32)
    for i in range(0, embeddings.shape[0]-1):
        out[i] = np.linalg.norm(embeddings[i] - embeddings[i+1], 2)
    return out

def main():
    use_cuda = torch.cuda.is_available()
    args = get_args()

    tcn = load_model(args.model)
    if use_cuda:
        tcn = tcn.cuda()

    video = read_video(args.test_video, (299, 299))
    video = Variable(torch.FloatTensor(video), volatile=True)

    embeddings = tcn(video)
    embeddings = embeddings.data.numpy()
    tsne = manifold.TSNE(perplexity=2, learning_rate=10)
    two_dim = tsne.fit_transform(embeddings)
    plot.scatter(two_dim[:, 0], two_dim[:, 1], c=np.linspace(0, 1, two_dim.shape[0]), cmap='jet', s=5)
    plot.show()


if __name__ == '__main__':
    main()
