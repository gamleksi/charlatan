import os
import numpy as np
import pybullet as bullet
import torch
from torch.autograd import Variable
from tcn import PosNet
from kuka.env import KukaPoseEnv
from util import read_video, ls, _resize_frame
from PIL import Image, ImageOps

class ImitationEnv(KukaPoseEnv):
    CAMERA_POS = {
        'distance': 5,
        'pitch': -20,
        'yaw': 0,
    }

    def __init__(self, video_dir=None, tcn=None, frame_size=None, transforms=transforms, **kwargs):
        self.video_index = -1
        self._build_video_paths(video_dir)
        self.use_cuda = torch.cuda.is_available()
        self.frame_size = frame_size
        super(ImitationEnv, self).__init__(**kwargs)
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 1e-3
        self.tcn = tcn
        if transforms is None:
            transforms = lambda x: x
        self.transforms = transforms

    def _build_video_paths(self, video_dir):
        video_paths = ls(video_dir)
        self.video_paths = [os.path.join(video_dir, f) for f in video_paths]

    def _reset(self):
        super(ImitationEnv, self)._reset()
        self.video_index = (self.video_index + 1) % len(self.video_paths)
        self.video = read_video(self.video_paths[self.video_index], self.frame_size)
        self.video_length = len(self.video)

    def _termination(self):
        return self._envStepCounter >= self.video_length

    def _get_current_frame(self):
        base_position, orientation = bullet.getBasePositionAndOrientation(self._kuka.kukaUid)
        view_matrix = bullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_position,
            distance=self.CAMERA_POS['distance'],
            yaw=self.CAMERA_POS['yaw'],
            pitch=self.CAMERA_POS['pitch'],
            roll=0,
            upAxisIndex=2)
        proj_matrix = bullet.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        _, _, px, _, _ = bullet.getCameraImage(
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            width=1000, height=900, renderer=bullet.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px)
        return _resize_frame(rgb_array, self.frame_size)

    def _reward(self):
        video_frame = self.transforms(self.video[self._envStepCounter])
        current_frame = self.transforms(self._get_current_frame())
        tensors = torch.stack(
            torch.Tensor(video_frame),
            torch.Tensor(current_frame)
            , dim=0)
        tensors = Variable(tensors, volatile=True)
        if self.use_cuda:
            tensors = tensors.cuda()
        embeddings = self.tcn(tensors).cpu().numpy()
        video_embedding = embeddings[0, :]
        frame_embedding = embeddings[1, :]
        distance = self._distance(video_embedding, frame_embedding)
        return (self.alpha *  distance - self.beta * np.sqrt(self.gamma + distance))

    def _distance(self, embedding1, embedding2):
        return np.power(embedding1 - embedding2, 2)

if __name__ == "__main__":
    frame_size = (299, 299)
    env = ImitationEnv(video_dir='./data/validation/angle1', tcn=lambda x: x, frame_size=frame_size)
    for _ in range(10):
        env.step(env.action_space.sample())


