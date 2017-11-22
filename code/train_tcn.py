import numpy as np
import torch
from torch import optim
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from util import VideoTripletDataset
from tcn import define_model

def distance(x1, x2):
    assert(x1.size() == x2.size())
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def batch_size(epoch):
    exponent = epoch // 3
    return min(max(2 ** (exponent), 1), 128)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()


def main():
    use_cuda = torch.cuda.is_available()

    tcn = define_model(use_cuda)

    dataset = VideoTripletDataset('./data/train/')
    args = get_args()

    for epoch in range(args.epochs):
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size(epoch),
            shuffle=False
        )

        optimizer = optim.SGD(tcn.parameters(), lr=1e-3, momentum=0.9)

        for minibatch in data_loader:
            frames = Variable(minibatch)

            if use_cuda:
                frames = frames.cuda()

            anchor_frames = frames[:, 0, :, :, :]
            positive_frames = frames[:, 1, :, :, :]
            negative_frames = frames[:, 2, :, :, :]

            positive_output = tcn(positive_frames)
            anchor_output = tcn(anchor_frames)
            negative_output = tcn(negative_frames)

            loss = distance(anchor_output, positive_output) - (
                distance(anchor_output, negative_output)
            )

            loss = torch.sum(loss)

            loss.backward()
            optimizer.step()
            print('loss: ', loss.data[0])


if __name__ == '__main__':
    main()