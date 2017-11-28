import os
import functools
import imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import RandomCrop
from torch import Tensor
from torch.autograd import Variable
import logging

def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def view_image(frame):
    # For debugging. Shows the image
    # Input shape (3, 299, 299) float32
    img = Image.fromarray(np.transpose(frame * 255, [1, 2, 0]).astype(np.uint8))
    img.show()

def _resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def read_video(filepath, frame_size):
    imageio_video = imageio.read(filepath)
    snap_length = len(imageio_video)
    frames = np.zeros((snap_length, 3, *frame_size))
    resized = map(lambda frame: _resize_frame(frame, frame_size), imageio_video)
    for i, frame in enumerate(resized):
        frames[i, :, :, :] = frame
    return frames

def ls_directories(path):
    return next(os.walk(path))[1]

def ls(path):
    # returns list of files in directory without hidden ones.
    return [p for p in os.listdir(path) if p[0] != '.']

class TripletBuilder(object):
    # The directory structure is such that the training directory contains several subfolders.
    # Each subfolder should contain distinct videos numbered 1-n such that they correspond to videos
    # in the sibling folders. E.g. video video_dir/angle1/1.mp4 should be of the same trajectory as
    # video_dir/angle2/1.mp4.
    def __init__(self, video_directory, image_size):
        self.frame_size = image_size
        self._read_angle_directories(video_directory)

        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # the frame.
        self.positive_frame_margin = 50
        self.video_index = 0

    def _read_angle_directories(self, video_directory):
        self._video_directory = video_directory
        self._angle_directories = [os.path.join(self._video_directory, p) for p in ls_directories(video_directory)]
        filenames = ls(self._angle_directories[0])
        self.angle1_paths = [os.path.join(self._angle_directories[0], f) for f in filenames]
        self.angle2_paths = [os.path.join(self._angle_directories[1], f) for f in filenames]
        self.video_count = len(self.angle1_paths)

    def _count_frames(self):
        frame_lengths1 = np.array([len(imageio.read(p)) for p in self.angle1_paths])
        frame_lengths2 = np.array([len(imageio.read(p)) for p in self.angle2_paths])
        self.frame_lengths = np.minimum(frame_lengths1, frame_lengths2)
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        snap1 = read_video(self.angle1_paths[index], self.frame_size)
        snap2 = read_video(self.angle2_paths[index], self.frame_size)
        return snap1, snap2

    def build_set(self, tcn):
        triplets = []
        for i in range(5):
            snap1, snap2 = self.get_video(self.video_index)
            anchor_frames, positive_frames, negative_frames = self.sample_frames(tcn, snap1, snap2)

            triplet_count = anchor_frames.size()[0]
            triplet = torch.zeros(triplet_count, 3, 3, *self.frame_size)
            triplet[:, 0, :, :, :] = anchor_frames
            triplet[:, 1, :, :, :] = positive_frames
            triplet[:, 2, :, :, :] = negative_frames
            triplets.append(triplet)
            self.video_index = (self.video_index + 1) % self.video_count
        tensors = torch.cat(triplets, dim=0)
        return TensorDataset(tensors, torch.zeros(tensors.size()[0]))

    def negative_frame_indices(self, positive_index, video_length):
        frame_range = np.arange(max(0, positive_index - self.positive_frame_margin),
            min(video_length, positive_index + self.positive_frame_margin))
        return frame_range

    def sample_frames(self, tcn, snap1, snap2):
        use_cuda = torch.cuda.is_available()
        positive_index = np.random.choice(np.arange(0, self.frame_lengths[self.video_index]))
        anchor_frame = Tensor(snap1[positive_index])
        positive_frame = Tensor(snap2[positive_index])
        negative_video = snap1 if np.random.rand() < 0.5 else snap2
        negative_video = Tensor(negative_video)

        negative_frame_indices = self.negative_frame_indices(positive_index, len(negative_video))
        np.random.shuffle(negative_frame_indices)
        if use_cuda:
            anchor_frame = anchor_frame.cuda()
            positive_frame = positive_frame.cuda()
            negative_video = negative_video.cuda()
        anchor_output = tcn(Variable(
            torch.unsqueeze(anchor_frame, dim=0),
            volatile=True))
        positive_output = tcn(Variable(
            torch.unsqueeze(positive_frame, dim=0),
            volatile=True))
        video_output = tcn(Variable(negative_video, volatile=True))
        positive_distances = distance(anchor_output, positive_output)
        negative_distances = distance(anchor_output, video_output)
        diff = (positive_distances < negative_distances).data.cpu().numpy()
        indices = np.where(diff == 1)[0]
        if indices.size == 0:
            indices = np.random.choice(negative_frame_indices, size=50)
        indices = torch.LongTensor(indices)
        if use_cuda:
            indices = indices.cuda()
        negative_frames = torch.index_select(negative_video, 0, indices)
        dim0size = negative_frames.size()[0]
        anchor_frame = anchor_frame.repeat(dim0size, 1, 1, 1)
        positive_frame = positive_frame.repeat(dim0size, 1, 1, 1)
        return anchor_frame, positive_frame, negative_frames

    def get_frame(self, video, index):
        return video[index]

class VideoTripletDataset(Dataset, TripletBuilder):
    def __len__(self):
        return sum(self.frame_lengths)

    def __getitem__(self, index):
        index = index
        video_index = self._video_index(index)
        angle1_video, angle2_video = self.get_video(video_index)

        frame_index, anchor_frame = self._get_anchor_frame(video_index, angle1_video, index)
        positive_frame = self._get_positive_frame(angle2_video, frame_index)

        negative_frame = self._get_negative_frame(angle1_video, angle2_video, frame_index)
        return np.stack([anchor_frame, positive_frame, negative_frame]).astype(np.float32)

    def _get_anchor_frame(self, video_index, video, index):
        frame_index = self._positive_frame_index(index, video_index)
        anchor_frame = self._read_frame(video, frame_index)
        return frame_index, anchor_frame

    def _get_positive_frame(self, video, frame_index):
        return self._read_frame(video, frame_index)

    def _get_negative_frame(self, angle1_video, angle2_video, positive_index):
        negative_video = angle1_video if np.random.rand() < 0.5 else angle2_video
        negative_frame_index = self._sample_negative_frame_index(negative_video, positive_index)
        return self._read_frame(negative_video, negative_frame_index)

    def _video_index(self, index):
        for i in range(self.video_count):
            if index <= self.cumulative_lengths[i]:
                return i

    def _positive_frame_index(self, index, video_index):
        if video_index == 0:
            frames_before_video = 0
        else:
            frames_before_video = self.cumulative_lengths[video_index-1]
        return index - frames_before_video - 1

    def _sample_negative_frame_index(self, video, positive_index):
        frames = len(video)
        return np.random.choice(self.negative_frame_indices(positive_index, frames))

    def _read_frame(self, video, frame_index):
        return video[frame_index]


class Logger(object):
    def __init__(self, logfilename):
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, filemode='w')

    def info(self, *arguments):
        print(*arguments)
        message = " ".join(map(repr, arguments))
        logging.info(message)
