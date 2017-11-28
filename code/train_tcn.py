import os
import numpy as np
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from util import VideoTripletDataset, TripletBuilder, distance, Logger
from tcn import define_model, PosNet
from torchvision import transforms


# Measured on validation set
color_means = [0.7274369 , 0.75985962, 0.83737165]
color_std = [0.01760677,  0.01127113, 0.00469666]
IMAGE_SIZE = (128, 128)

normalize = transforms.Normalize(color_means, color_std)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/')
    parser.add_argument('--train-directory', type=str, default='./data/train/')
    parser.add_argument('--validation-directory', type=str, default='./data/validation/')
    parser.add_argument('--max-minibatch-size', type=int, default=64)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--model-name', type=str, default='tcn')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.01)
    return parser.parse_args()

arguments = get_args()

logger = Logger(arguments.log_file)
def batch_size(epoch, max_size):
    exponent = epoch // 10
    return min(max(2 ** (exponent), 1), max_size)

def validate(tcn, use_cuda, arguments):
    # Run model on validation data and log results
    dataset = VideoTripletDataset(arguments.validation_directory, IMAGE_SIZE)
    data_loader = DataLoader(dataset, batch_size=arguments.max_minibatch_size, shuffle=False, num_workers=2)

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

    logger.info("Validation score correct: {0}/{1}".format(num_correct, len(dataset)))


def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def model_filename(model_name, epoch):
    return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

def save_model(model, filename, model_folder):
    ensure_folder(model_folder)
    model_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), model_path)


def main():
    use_cuda = torch.cuda.is_available()

    # tcn = define_model(use_cuda)
    tcn = PosNet()
    if use_cuda:
        tcn.cuda()

    dataset = VideoTripletDataset(arguments.train_directory, IMAGE_SIZE)

    optimizer = optim.SGD(tcn.parameters(), lr=arguments.lr_start, momentum=0.9)

    for epoch in range(1, arguments.epochs):
        logger.info("Starting epoch: {0}".format(epoch))
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size(epoch, arguments.max_minibatch_size),
            num_workers=2
        )

        if epoch % 10 == 0:
            validate(tcn, use_cuda, arguments)

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
            loss = torch.clamp(arguments.margin + d_positive - d_negative, min=0.0).mean()

            loss.backward()
            optimizer.step()

        logger.info('loss: ', loss.data[0])

        if epoch % arguments.save_every == 0:
            logger.info('Saving model.')
            save_model(tcn, model_filename(arguments.model_name, epoch), arguments.model_folder)





if __name__ == '__main__':
    main()
