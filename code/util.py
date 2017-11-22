import os
import functools
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def _resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image) / 255
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

class VideoTripletDataset(Dataset):
    # The directory structure is such that the training directory contains several subfolders.
    # Each subfolder should contain distinct videos numbered 1-n such that they correspond to videos
    # in the sibling folders. E.g. video video_dir/angle1/1.mp4 should be of the same trajectory as
    # video_dir/angle2/1.mp4.
    def __init__(self, video_directory):
        self.frame_size = (299, 299)
        self._read_angle_directories(video_directory)

        # Note: Corresponding videos must have same frame count.
        self._count_frames()
        # The negative example has to be from outside the buffer window. Taken from both sides of
        # the frame.
        self.positive_frame_margin = 24
        self.multiple = 5

    def _read_angle_directories(self, video_directory):
        self._video_directory = video_directory
        self._angle_directories = [os.path.join(self._video_directory, p) for p in ls_directories(video_directory)]
        filenames = ls(self._angle_directories[0])
        self.angle1_paths = [os.path.join(self._angle_directories[0], f) for f in filenames]
        self.angle2_paths = [os.path.join(self._angle_directories[1], f) for f in filenames]
        self.video_count = len(self.angle1_paths)

    def _count_frames(self):
        self.frame_lengths = np.array([len(imageio.read(p)) for p in self.angle1_paths])
        self.cumulative_lengths = np.zeros(len(self.frame_lengths))
        for i, frames in enumerate(self.frame_lengths):
            if i == 0:
                prev = 0
            else:
                prev = self.frame_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    @functools.lru_cache(maxsize=3)
    def get_video(self, index):
        snap1 = read_video(self.angle1_paths[index], self.frame_size)
        snap2 = read_video(self.angle2_paths[index], self.frame_size)
        return snap1, snap2

    def __len__(self):
        # Technically we could construct almost n * n * n triplets where n is the
        # amount of frames we have. The multiple is a somewhat arbitrary setting.
        return sum(self.frame_lengths) * self.multiple

    def __getitem__(self, index):
        index = index % (len(self) // self.multiple)
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
            if index < self.cumulative_lengths[i]:
                return i

    def _positive_frame_index(self, index, video_index):
        if video_index == 0:
            frames_before_video = 0
        else:
            frames_before_video = self.cumulative_lengths[video_index-1]
        return index - frames_before_video

    def _sample_negative_frame_index(self, video, positive_index):
        frames = len(video)
        frame_range = np.arange(max(0, positive_index - self.positive_frame_margin),
            min(frames, positive_index + self.positive_frame_margin)
            , 1)
        return np.random.choice(frame_range)

    def _read_frame(self, video, frame_index):
        return video[frame_index]
