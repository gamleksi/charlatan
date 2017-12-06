import os
import numpy as np
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from util import (VideoTripletDataset, SingleViewTripletBuilder, distance, Logger, ensure_folder,
            normalize)
from tcn import define_model, PosNet

IMAGE_SIZE = (128, 128)

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
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    return parser.parse_args()

arguments = get_args()

logger = Logger(arguments.log_file)
def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 1), max_size)

validation_builder = SingleViewTripletBuilder(arguments.validation_directory, IMAGE_SIZE, arguments, look_for_negative=False)
validation_set = validation_builder.build_set()
del validation_builder

def validate(tcn, use_cuda, arguments):
    # Run model on validation data and log results
    data_loader = DataLoader(validation_set, batch_size=arguments.max_minibatch_size, shuffle=False)
    num_correct = 0
    for minibatch, _ in data_loader:
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

        num_correct += ((d_positive + arguments.margin) < d_negative).data.cpu().numpy().sum()

    logger.info("Validation score correct: {0}/{1}".format(num_correct, len(validation_set)))

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

    triplet_builder = SingleViewTripletBuilder(arguments.train_directory, IMAGE_SIZE, arguments)

    optimizer = optim.SGD(tcn.parameters(), lr=arguments.lr_start, momentum=0.9)

    for epoch in range(0, arguments.epochs):
        logger.info("Starting epoch: {0}".format(epoch))
        dataset = triplet_builder.build_set(tcn)
        logger.info("Created {0} triplets".format(len(dataset)))
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size(epoch, arguments.max_minibatch_size),
            shuffle=True
        )

        if epoch % 10 == 0:
            validate(tcn, use_cuda, arguments)

        for minibatch, _ in data_loader:
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

        if epoch % arguments.save_every == 0 and epoch != 0:
            logger.info('Saving model.')
            save_model(tcn, model_filename(arguments.model_name, epoch), arguments.model_folder)





if __name__ == '__main__':
    main()
