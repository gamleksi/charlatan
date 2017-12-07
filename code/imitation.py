import os
import numpy as np
import pybullet as bullet
import torch
from torch.autograd import Variable
from tcn import PosNet
from kuka.env import KukaPoseEnv
from util import read_video, ls, _resize_frame
from PIL import Image, ImageOps
from gym import spaces

class ImitationEnv(KukaPoseEnv):
    CAMERA_POS = {
        'distance': 5,
        'pitch': -20,
        'yaw': 0,
    }

    def __init__(self, video_dir=None, tcn=None, frame_size=None, transforms=None, num_embedding_observations=1, **kwargs):
        self.video_index = -1
        self._build_video_paths(video_dir)
        self.use_cuda = torch.cuda.is_available()
        self.frame_size = frame_size
        self.num_embedding_observations = num_embedding_observations
        self.tcn = tcn
        if transforms is None:
            transforms = lambda x: x
        self.transforms = transforms
        self.initialize_video_data(0)
        super(ImitationEnv, self).__init__(**kwargs)
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 1e-3
        self._setup_spaces()

    def _build_video_paths(self, video_dir):
        video_paths = ls(video_dir)
        self.video_paths = [os.path.join(video_dir, f) for f in video_paths]

    def _reset(self):
        self.video_index = (self.video_index + 1) % len(self.video_paths)
        self.initialize_video_data(self.video_index)
        observation = super(ImitationEnv, self)._reset()
        return observation
    
    def initialize_video_data(self, video_index):
        self.video = read_video(self.video_paths[video_index], self.frame_size)
        self.video_length = len(self.video)


    def _setup_goal_space(self):
        embedding_space = self.embedding_observations()
        embedding_space = embedding_space.flatten()
        self.goal_space = spaces.Box(
            low=embedding_space[0],
            high=embedding_space[1])

    def _setup_observation_space(self):
        embedding_space = self.embedding_observations()
        embedding_space = embedding_space.flatten()
        self.observation_space = spaces.Box(
            low=np.concatenate((self.joint_lower_limit, embedding_space)),
            high=np.concatenate((self.joint_upper_limit, embedding_space)))


    def embedding_observations(self):
        video_frames = self.video[self._envStepCounter]
        video_frames = self.transforms(torch.Tensor(video_frames))
        embeddings = self.frame_embeddings([video_frames])
        return embeddings


    def frame_embeddings(self, frames):
        tensors = torch.stack(
            frames, dim=0
            )
        tensors = Variable(tensors, volatile=True)
        if self.use_cuda:
            tensors = tensors.cuda()
        embeddings = self.tcn(tensors).cpu().data.numpy()
        return embeddings

    def _termination(self):
        return self._envStepCounter >= self.video_length - 1

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

        rgb_array = np.array(px[:,:,0:3])
        rgb_array = _resize_frame(rgb_array, self.frame_size)
        return rgb_array

    def buildObservation(self):
        embeddings = self.embedding_observations()
        observation = np.concatenate((self._observation, embeddings.flatten()))
        return observation

    def debug_image(self, frame,id):
        import cv2
        cv2.imwrite('state-{}.png'.format(id), frame)

    def _reward(self):
        video_frame = torch.Tensor(self.video[self._envStepCounter])
        current_frame = torch.Tensor(self._get_current_frame())
        embeddings = self.frame_embeddings([
            self.transforms(video_frame),
            self.transforms(current_frame)]) 
        video_embedding = embeddings[0, :]
        frame_embedding = embeddings[1, :]
        distance = self._distance(video_embedding, frame_embedding)
        return (- self.alpha *  distance - self.beta * np.sqrt(self.gamma + distance))

    def _distance(self, embedding1, embedding2):
        return np.sum(np.power(embedding1 - embedding2, 2))

from util import normalize, view_image
from tcn import define_model

if __name__ == "__main__":
    frame_size = (299, 299)
    
    tcn = define_model(False)
    model_path = os.path.join(
        "./trained_models/tcn",
        "inception-epoch-2000.pk"
    )
    tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    transforms = normalize

    env = ImitationEnv(transforms=transforms, renders=False, video_dir='./data/video/angle-1', tcn=tcn, frame_size=frame_size)
    done = False
    while not(done):
        observations, reward, done, _ = env.step(env.action_space.sample())