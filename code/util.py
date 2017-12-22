import os
import functools
import imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
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

def write_to_csv(values, keys, filepath):
    if  not(os.path.isfile(filepath)):
        with open(filepath, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open(filepath, 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)


def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def _resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def write_video(file_name, path, frames):
    imageio.mimwrite(os.path.join(path, file_name), frames, fps=60)

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

class SingleViewTripletBuilder(object):
    def __init__(self, video_directory, image_size, cli_args, sample_size=500):
        self.frame_size = image_size
        self._read_angle_directories(video_directory)

        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.positive_frame_margin = 10
        self.negative_frame_margin = 30
        self.video_index = 0
        self.cli_args = cli_args
        self.sample_size = sample_size

    def _read_angle_directories(self, video_directory):
        self._video_directory = video_directory
        self._angle_directories = [os.path.join(self._video_directory, p) for p in ls_directories(video_directory)]
        filenames = ls(self._angle_directories[0])
        self.angle1_paths = [os.path.join(self._angle_directories[0], f) for f in filenames]
        self.video_count = len(self.angle1_paths)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.angle1_paths])
        self.frame_lengths = frame_lengths
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        return read_video(self.angle1_paths[index], self.frame_size)

    def sample_triplet(self, snap):
        anchor_index = self.sample_anchor_frame_index()
        positive_index = self.sample_positive_frame_index(anchor_index)
        negative_index = self.sample_negative_frame_index(anchor_index)
        anchor_frame = snap[anchor_index]
        positive_frame = snap[positive_index]
        negative_frame = snap[negative_index]
        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
        for i in range(0, self.sample_size):
            snap = self.get_video(self.video_index)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snap)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame

        self.video_index = (self.video_index + 1) % self.video_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def sample_anchor_frame_index(self):
        arange = np.arange(0, self.frame_lengths[self.video_index])
        return np.random.choice(arange)

    def sample_positive_frame_index(self, anchor_index):
        lower_bound = max(0, anchor_index - self.positive_frame_margin)
        range1 = np.arange(lower_bound, anchor_index)
        upper_bound = min(self.frame_lengths[self.video_index] - 1, anchor_index + self.positive_frame_margin)
        range2 = np.arange(anchor_index + 1, upper_bound)
        return np.random.choice(np.concatenate([range1, range2]))

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.video_index]
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))

class MultiViewTripletBuilder(SingleViewTripletBuilder):
    # The directory structure is such that the training directory contains several subfolders.
    # Each subfolder should contain distinct videos numbered 1-n such that they correspond to videos
    # in the sibling folders. E.g. video video_dir/angle1/1.mp4 should be of the same trajectory as
    # video_dir/angle2/1.mp4.
    def _read_angle_directories(self, video_directory):
        self._video_directory = video_directory
        self._angle_directories = [os.path.join(self._video_directory, p) for p in ls_directories(video_directory)]
        filenames = ls(self._angle_directories[0])
        self.angle1_paths = [os.path.join(self._angle_directories[0], f) for f in filenames]
        self.angle2_paths = [os.path.join(self._angle_directories[1], f) for f in filenames]
        self.video_count = len(self.angle1_paths)

    def _count_frames(self):
        super()._count_frames()
        frame_lengths2 = np.array([len(imageio.read(p)) for p in self.angle2_paths])
        self.frame_lengths = np.minimum(self.frame_lengths, frame_lengths2)
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    def build_set(self, tcn):
        triplets = []
        for i in range(self.cli_args.triplets_from_videos):
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

    @functools.lru_cache(maxsize=1)
    def get_video(self, index):
        snap1 = read_video(self.angle1_paths[index], self.frame_size)
        snap2 = read_video(self.angle2_paths[index], self.frame_size)
        return snap1, snap2

    def sample_frames(self, tcn, snap1, snap2):
        use_cuda = torch.cuda.is_available()

        positive_index = np.random.choice(np.arange(0, self.frame_lengths[self.video_index]))
        anchor_frame = Tensor(snap1[positive_index])
        positive_frame = Tensor(snap2[positive_index])
        negative_video = snap1 if np.random.rand() < 0.5 else snap2
        negative_frames = self.find_negative_frames(tcn, negative_video, positive_index, anchor_frame, positive_frame)

        dim0size = negative_frames.size()[0]
        anchor_frame = anchor_frame.repeat(dim0size, 1, 1, 1)
        positive_frame = positive_frame.repeat(dim0size, 1, 1, 1)
        return anchor_frame, positive_frame, negative_frames

class VideoTripletDataset(Dataset, MultiViewTripletBuilder):
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
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, filemode='a')

    def info(self, *arguments):
        print(*arguments)
        message = " ".join(map(repr, arguments))
        logging.info(message)
