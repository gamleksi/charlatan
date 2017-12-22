import os
import numpy as np
import pybullet as bullet
import torch
from torch.autograd import Variable
from tcn import PosNet
from kuka.env import KukaSevenJointsEnv # KukaPoseEnv
from util import read_video, ls, _resize_frame
from PIL import Image, ImageOps
from gym import spaces
from tcn import define_model

class ImitationEnv(KukaSevenJointsEnv):
    CAMERA_POS = {
        'distance': 5,
        'pitch': -20,
        'yaw': 0,
    }

    def __init__(self, video_dir=None, frame_size=None, **kwargs):
        self.video_index = -1
        self._frame_counter = 0
        self._frames_repeated = 0
        self._build_video_paths(video_dir)
        self.frame_size = frame_size
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
        print("reset")
        self._frame_counter = 0
        self._frames_repeated = 1
        self.video_index = (self.video_index + 1) % len(self.video_paths)
        self.initialize_video_data(self.video_index)
        observation = super(ImitationEnv, self)._reset()
        return observation

    def initialize_video_data(self, video_index):
        self.video = read_video(self.video_paths[video_index], self.frame_size)
        self.video_length = len(self.video)

    def _setup_observation_space(self):
        embedding_space = np.zeros(32)
        print(embedding_space.shape)
        self.observation_space = spaces.Box(
            low=np.concatenate((self.joint_lower_limit, -self.joint_velocity_limit, embedding_space)),
            high=np.concatenate((self.joint_upper_limit, self.joint_velocity_limit, embedding_space)))


    def frame_embeddings(self, frames):
        tensors = Variable(frames, volatile=True)
        if self.use_cuda:
            tensors = tensors.cuda(0)
        embeddings = self.tcn(tensors).cpu().data.numpy()
        return embeddings

    def _termination(self):

        if self._frames_repeated < 3:
            self._frames_repeated += 1
        else:
            self._frames_repeated = 0
            self._frame_counter += 1

        return self.video_length - 1 < self._frame_counter

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

    def _step(self, action):
        action = action * self._kuka.maxForce
        self._kuka.applyAction(action)
        bullet.stepSimulation()
        if self._renders:
            time.sleep(self._timeStep)
        reward = self._reward()
        self._envStepCounter += 1
        done = self._termination()
        if not(done):
            self._observation = self.getExtendedObservation()
        return self._observation, reward, done, None

    def _reward(self):
        video_frame = self.video[self._frame_counter]
        current_frame = self._get_current_frame()
        return [video_frame, current_frame]

    def _distance(self, embedding1, embedding2):
        return np.sum(np.power(embedding1 - embedding2, 2), axis=1)

    def getExtendedObservation(self):
        next_frame = self.video[self._frame_counter]
        joint_positions = self._motorized_joint_positions()
        joint_velocities = self._joint_velocities()
        return [np.concatenate((joint_positions, joint_velocities)), np.array([next_frame])]


if __name__ == "__main__":
    frame_size = (299, 299)

    tcn = define_model(False)
    model_path = os.path.join(
        "./trained_models/tcn",
        "inception-epoch-2000.pk"
    )
    tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    env = ImitationEnv(renders=True, video_dir='./data/validation/angle1', tcn=tcn, frame_size=frame_size)
    done = False
    while not(done):
        observations, reward, done, _ = env.step(env.action_space.sample())

        assert len(x.size()) == 4