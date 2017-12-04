import os
import numpy as np
import argparse
import torch
from torch import optim
from torch import multiprocessing
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms
from util import (VideoTripletDataset, SingleViewTripletBuilder, distance, Logger, ensure_folder,
            normalize)
from tcn import define_model, PosNet

IMAGE_SIZE = (299, 299)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--train-directory', type=str, default='./data/train/')
    parser.add_argument('--validation-directory', type=str, default='./data/validation/')
    parser.add_argument('--minibatch-size', type=int, default=256)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--model-name', type=str, default='tcn')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.01)
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    return parser.parse_args()

arguments = get_args()

class RandomNoiseTransform(object):
    def __init__(self, scale=0.2):
        self.scale = 0.2

    def __call__(self, triplet_minibatch):
        noise = torch.normal(0.0,
            torch.Tensor(np.ones(triplet_minibatch.size()) * self.scale)
        )
        return triplet_minibatch.add_(noise)

logger = Logger(arguments.log_file)
def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 2), max_size)

validation_builder = SingleViewTripletBuilder(arguments.validation_directory, IMAGE_SIZE, arguments,
    look_for_negative=False,
    transforms=normalize)
validation_set = validation_builder.build_set()
del validation_builder

def validate(tcn, use_cuda, arguments):
    # Run model on validation data and log results
    data_loader = DataLoader(validation_set, batch_size=1024, shuffle=False, pin_memory=True)
    num_correct = 0
    for minibatch, _ in data_loader:
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


def build_set(queue, triplet_builder, log):
    while 1:
        dataset = triplet_builder.build_set()
        log.info('Created {0} triplets'.format(len(dataset)))
        queue.put(dataset)

def create_model(use_cuda):
    tcn = define_model(use_cuda)
    # tcn = PosNet()
    if arguments.load_model:
        model_path = os.path.join(
            arguments.model_folder,
            arguments.load_model
        )
        # map_location allows us to load models trained on cuda to cpu.
        tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        tcn = tcn.cuda()
    return tcn


def main():
    use_cuda = torch.cuda.is_available()

    tcn = create_model(use_cuda)

    training_transforms = transforms.Compose([
        normalize,
        RandomNoiseTransform(scale=0.2)
    ])

    triplet_builder = SingleViewTripletBuilder(arguments.train_directory, IMAGE_SIZE, arguments, look_for_negative=False,
        transforms=training_transforms)

    queue = multiprocessing.Queue(3)
    dataset_builder_process = multiprocessing.Process(target=build_set, args=(queue, triplet_builder, logger), daemon=True)
    dataset_builder_process.start()

    optimizer = optim.SGD(tcn.parameters(), lr=arguments.lr_start, momentum=0.9)
    # This will diminish the learning rate at the milestones.
    # 0.1, 0.01, 0.001
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000], gamma=0.1)

    ITERATE_OVER_TRIPLETS = 5

    for epoch in range(arguments.start_epoch, arguments.start_epoch + arguments.epochs):
        logger.info("Starting epoch: {0} learning rate: {1}".format(epoch,
            learning_rate_scheduler.get_lr()))
        learning_rate_scheduler.step()

        dataset = queue.get()
        logger.info("Got {0} triplets".format(len(dataset)))

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=arguments.minibatch_size, # batch_size(epoch, arguments.max_minibatch_size),
            shuffle=False,
            pin_memory=True
        )

        if epoch % 10 == 0:
            validate(tcn, use_cuda, arguments)

        for i in range(0, ITERATE_OVER_TRIPLETS):
            losses = []
            for minibatch, _ in data_loader:
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

                losses.append(loss.numpy()[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info('loss: ', np.mean(losses))

        if epoch % arguments.save_every == 0 and epoch != 0:
            logger.info('Saving model.')
            save_model(tcn, model_filename(arguments.model_name, epoch), arguments.model_folder)





if __name__ == '__main__':
    main()
