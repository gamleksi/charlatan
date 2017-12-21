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
from util import normalize as transforms
from tcn import define_model

class ImitationEnv(KukaSevenJointsEnv):
    CAMERA_POS = {
        'distance': 5,
        'pitch': -20,
        'yaw': 0,
    }

    def __init__(self, video_dir=None, tcn=None, frame_size=None, transforms=None, num_embedding_observations=1, use_cuda=torch.cuda.is_available(), **kwargs):
        self.video_index = -1
        self._frame_counter = 0
        self._frames_repeated = 0 
        self._build_video_paths(video_dir)
        self.use_cuda = use_cuda  
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

# No need
#    def _setup_goal_space(self):
#        embedding_space = self.embedding_observations()
#        embedding_space = embedding_space.flatten()
#        self.goal_space = spaces.Box(
#            low=embedding_space[0],
#            high=embedding_space[1])

    def _setup_observation_space(self):
        #embedding_space = self.embedding_observations()
        # embedding_space = embedding_space.flatten()
        embedding_space = np.zeros(32)
        print(embedding_space.shape)
        self.observation_space = spaces.Box(
            low=np.concatenate((self.joint_lower_limit, -self.joint_velocity_limit, embedding_space)),
            high=np.concatenate((self.joint_upper_limit, self.joint_velocity_limit, embedding_space)))

    def embedding_observations(self):
        video_frames = self.video[self._frame_counter]
        video_frames = self.transforms(torch.Tensor(video_frames))
        embeddings = self.frame_embeddings([video_frames])
        return embeddings

    def frame_embeddings(self, frames):
        tensors = torch.stack(
            frames, dim=0
            )
        tensors = Variable(tensors, volatile=True)
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

    def _reward(self):
        video_frame = torch.Tensor(self.video[self._frame_counter])
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

    def getExtendedObservation(self):
        embeddings = self.embedding_observations()
        joint_positions = self._motorized_joint_positions()
        joint_velocities = self._joint_velocities()
        return np.concatenate((joint_positions, joint_velocities, embeddings.flatten()))

class ImitationTestEnv(ImitationEnv):
    def __init__(self, **kwargs):
        super(ImitationTestEnv, self).__init__(**kwargs)
    
    def _step(self, action):
        observation, reward, done, _ = super(ImitationTestEnv, self)._step(action)
        return observation, reward, done, self._get_current_frame() 


class ImitationWrapperEnv(ImitationEnv):
    def __init__(self, **kwargs):
        super(ImitationWrapperEnv, self).__init__(**kwargs)

    def _reward(self):
        video_frame = self.video[self._frame_counter]
        current_frame = self._get_current_frame()
        return [video_frame, current_frame]

    def getExtendedObservation(self):
        next_frame = self.video[self._frame_counter]
        joint_positions = self._motorized_joint_positions()
        joint_velocities = self._joint_velocities()
        return [np.concatenate((joint_positions, joint_velocities)), np.array([next_frame])]

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
        return self._observation, reward, done, {}


from tcn import define_model

class TCNWrapperEnv(ImitationEnv):
    def __init__(self, model_path="./trained_models/tcn", model="inception-epoch-2000.pk"):
        self.use_cuda = torch.cuda.is_available()
        self.tcn = self.load_model(model_path, model)
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 1e-3

    def load_model(self, model_path, model):
        tcn = define_model(self.use_cuda)
        model_path = os.path.join(
            model_path,
            model 
        )
        tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        if self.use_cuda:
            tcn = tcn.cuda(0)
        return tcn

    def frame_embeddings(self, frames):
        frames = torch.Tensor(frames)
        for idx in range(frames.shape[0]):
            frames[idx] = transforms(frames[idx])
        embeddings = super(TCNWrapperEnv,self).frame_embeddings(frames)
        return embeddings

    def reward(self, video_frames, current_frames):
        frames = np.concatenate((video_frames, current_frames))
        frame_embeddings = self.frame_embeddings(
            frames
            ) 
        video_embeddings = frame_embeddings[:video_frames.shape[0]]
        current_embeddings = frame_embeddings[video_frames.shape[0]:]
        distance = self._distance(video_embeddings, current_embeddings)
        return (- self.alpha *  distance - self.beta * np.sqrt(self.gamma + distance))

    def _distance(self, embedding1, embedding2):
        return np.sum(np.power(embedding1 - embedding2, 2), axis=1)

    def reward2(self, video_frames, current_frames):
        video_embeddings = self.frame_embeddings(
            video_frames
            )
        current_embeddings = self.frame_embeddings(
            current_frames 
            )
        distance = self._distance(video_embeddings, current_embeddings)
        return (- self.alpha *  distance - self.beta * np.sqrt(self.gamma + distance))


if __name__ == "__main__":
    frame_size = (299, 299)

    tcn = define_model(False)
    model_path = os.path.join(
        "./trained_models/tcn",
        "inception-epoch-2000.pk"
    )
    tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    transforms = normalize

    env = ImitationEnv(transforms=transforms, renders=True, video_dir='./data/video/angle-1', tcn=tcn, frame_size=frame_size)
    done = False
    while not(done):
        observations, reward, done, _ = env.step(env.action_space.sample())
