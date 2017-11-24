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
from torchvision import transforms

MARGIN = 0.5

# Measured on validation set
color_means = [0.7274369 , 0.75985962, 0.83737165]
color_std = [0.01760677,  0.01127113, 0.00469666]

normalize = transforms.Normalize(color_means, color_std)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/')
    parser.add_argument('--train-directory', type=str, default='./data/train/')
    parser.add_argument('--validation-directory', type=str, default='./data/validation/')
    parser.add_argument('--max-minibatch-size', type=int, default=64)
    return parser.parse_args()

def distance(x1, x2):
    assert(x1.size() == x2.size())
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def batch_size(epoch, max_size):
    exponent = epoch // 10
    return min(max(2 ** (exponent), 1), max_size)

def validate(tcn, use_cuda, arguments):
    # Run model on validation data and print results
    dataset = VideoTripletDataset(arguments.validation_directory)
    data_loader = DataLoader(dataset, batch_size=arguments.max_minibatch_size, shuffle=False)

    means = []
    stds = []

    num_correct = 0
    for minibatch in data_loader:
        minibatch = normalize(minibatch)
        frames = Variable(minibatch, volatile=True)

        if use_cuda:
            frames = frames.cuda()

        anchor_frames = frames[:, 0, :, :, :]
        positive_frames = frames[:, 1, :, :, :]
        negative_frames = frames[:, 2, :, :, :]

        anchor_output = tcn(anchor_frames)
        positive_output = tcn(positive_frames)
        negative_output = tcn(negative_frames)

        d_positive = distance(anchor_output, positive_output)
        d_negative = distance(anchor_output, negative_output)

        assert(d_positive.size()[0] == minibatch.size()[0])

        num_correct += (d_positive < d_negative).data.cpu().numpy().sum()

    print("Validation score correct: {0}/{1}".format(num_correct, len(dataset)))


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

    arguments = get_args()
    dataset = VideoTripletDataset(arguments.train_directory)

    for epoch in range(1, arguments.epochs):
        print("Starting epoch: {0}".format(epoch))
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size(epoch, arguments.max_minibatch_size)
        )

        if epoch % 10 == 0:
            validate(tcn, use_cuda, arguments)

        optimizer = optim.SGD(tcn.parameters(), lr=1e-3, momentum=0.9)

        for minibatch in data_loader:
            minibatch = normalize(minibatch)
            frames = Variable(minibatch)

            if use_cuda:
                frames = frames.cuda()

            anchor_frames = frames[:, 0, :, :, :]
            positive_frames = frames[:, 1, :, :, :]
            negative_frames = frames[:, 2, :, :, :]

            anchor_output = tcn(anchor_frames)
            positive_output = tcn(positive_frames)
            negative_output = tcn(negative_frames)

            d_positive = distance(anchor_output, positive_output)
            d_negative = distance(anchor_output, negative_output)
            loss = torch.clamp(MARGIN + d_positive - d_negative, min=0.0).mean()

            loss.backward()
            optimizer.step()

        print('loss: ', loss.data[0])

        if epoch % arguments.save_every == 0:
            print('Saving model.')
            save_model(tcn, model_filename(epoch), arguments.model_folder)





if __name__ == '__main__':
    main()
