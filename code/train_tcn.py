import os
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
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/')
    parser.add_argument('--train-directory', type=str, default='./data/train/')
    parser.add_argument('--validation-directory', type=str, default='./data/validation/')
    return parser.parse_args()

def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def model_filename(epoch):
    return "tcn-epoch-{0}.pk".format(epoch)

def save_model(model, filename, model_folder):
    ensure_folder(model_folder)
    model_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), model_path)

def main():
    use_cuda = torch.cuda.is_available()

    tcn = define_model(use_cuda)

    args = get_args()
    dataset = VideoTripletDataset(args.train_directory)

    for epoch in range(1, args.epochs):
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

        if epoch % args.save_every == 0:
            save_model(tcn, model_filename(epoch), args.model_folder)


if __name__ == '__main__':
    main()
