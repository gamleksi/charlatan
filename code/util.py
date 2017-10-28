import os
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class VideoTripletDataset(Dataset):
    def __init__(self, video_directory):
        self._video_dir = video_directory
        self.videos = [self._read_video(v) for v in os.listdir(video_directory)]
        self.frame_lengths = np.array([len(v) for v in self.videos])
        self.cumulative_lengths = np.zeros(len(self.frame_lengths))
        for i, frames in enumerate(self.frame_lengths):
            if i == 0:
                prev = 0
            else:
                prev = self.frame_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

        # The negative example has to be from outside the buffer window. Taken from both sides of
        # the frame.
        self.positive_frame_margin = 24
        self.multiple = 5

    def _read_video(self, path):
        path = os.path.join(self._video_dir, path)
        return imageio.read(path)

    def __len__(self):
        # Technically we could construct almost n * n * n triplets where n is the
        # amount of frames we have. The multiple is a somewhat arbitrary setting.
        return sum(self.frame_lengths) * self.multiple

    def __getitem__(self, index):
        index = index % (len(self) // self.multiple)
        video_index = self._video_index(index)
        video = self.videos[video_index]

        positive_frame_index = self._positive_frame_index(index, video_index)
        positive_frame = self._read_frame(video, positive_frame_index)
        # The anchor frame should be a frame from the same point in time from a different angle.
        # How this will be implemented will be decided later.
        anchor_frame = positive_frame

        negative_frame_index = self._sample_negative_frame_index(video_index, positive_frame_index)
        negative_frame = self._read_frame(video, negative_frame_index)
        return self._process_frames(positive_frame, anchor_frame, negative_frame)

    def _process_frames(self, *frames):
        resized = []
        for frame in frames:
            image = Image.fromarray(frame)
            image = image.resize((299, 299))
            resized.append(np.array(image) / 255)

        stacked = np.stack(resized).astype(np.float32)
        return np.transpose(stacked, [0, 3, 1, 2])

    def _video_index(self, index):
        for i in range(len(self.videos)):
            if index < self.cumulative_lengths[i]:
                return i

    def _positive_frame_index(self, index, video_index):
        if video_index == 0:
            frames_before_video = 0
        else:
            frames_before_video = self.cumulative_lengths[video_index-1]
        return index - frames_before_video

    def _sample_negative_frame_index(self, video_index, positive_index):
        frame_range = np.arange(max(0, positive_index - self.positive_frame_margin),
            min(self.frame_lengths[video_index], positive_index + self.positive_frame_margin)
            , 1)
        return np.random.choice(frame_range)

    def _read_frame(self, video, frame_index):
        return video.get_data(frame_index)
