import os
import gym
from imitation import ImitationEnv
import numpy as np
import torch
from torch.autograd import Variable
from tcn import define_model


class RewardHelper(object):
    def __init__(self):
        self.alpha = 0.5
        self.beta = 0.5
        self.gamma = 1e-3
        self.use_cuda = torch.cuda.is_available()
        self.tcn = self.load_model('./trained_models/tcn/', 'inception-epoch-2000.pk')

    def load_model(self, model_path, model):
        tcn = define_model(self.use_cuda)
        model_path = os.path.join(
            model_path,
            model
        )
        tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        return tcn

    def frame_embeddings(self, frames):
        frames = Variable(frames, volatile=True)
        if self.use_cuda:
            frames = frames.cuda()
        return self.tcn(frames).data.cpu().numpy()

    def build_frame_embeddings(self, frames):
        frames = torch.Tensor(frames)
        return self.frame_embeddings(frames)

    def huber_loss(self, distance):
        return -self.alpha * distance - self.beta * np.sqrt(self.gamma + distance)

    def reward(self, video_frames, current_frames):
        frames = torch.cat([torch.Tensor(video_frames), torch.Tensor(current_frames)], dim=0)
        assert len(frames.size()) == 4
        frame_embeddings = self.frame_embeddings(frames)
        assert frame_embeddings.shape == (frames.shape[0], 32)
        video_embeddings = frame_embeddings[0:video_frames.shape[0]]
        current_embeddings = frame_embeddings[video_frames.shape[0]:]
        distance = self._distance(video_embeddings, current_embeddings)
        assert video_embeddings.shape == (video_frames.shape[0], 32)
        assert current_embeddings.shape == (current_frames.shape[0], 32)
        assert distance.shape == (video_frames.shape[0],)
        return self.huber_loss(distance)

    def _distance(self, embedding1, embedding2):
        assert embedding1.shape == embedding2.shape
        assert len(embedding1.shape) == 2
        return np.sum(np.power(embedding1 - embedding2, 2), axis=1)

    def reward2(self, video_frames, current_frames):
        video_embeddings = self.frame_embeddings(
            torch.Tensor(video_frames)
            )
        current_embeddings = self.frame_embeddings(
            torch.Tensor(current_frames)
            )
        assert video_embeddings.shape == (1, 32)
        assert current_embeddings.shape == (1, 32)
        distance = self._distance(video_embeddings, current_embeddings)
        assert distance.shape == (video_embeddings.shape[0],)
        return self.huber_loss(distance)


def main():
    env = gym.make('KukaImitationTest-v0')
    env.seed(2222)
    done = False
    steps = 0
    reward_helper = RewardHelper()

    while not(done):
        observation, reward, done, _ = env.step(env.action_space.sample())
        video_frames = reward[0]
        current_frames = reward[1]
        import ipdb; ipdb.set_trace()
        reward = reward_helper.reward2(video_frames[None], current_frames[None])
        print('reward 1', reward)
        reward = reward_helper.reward(video_frames[None], current_frames[None])
        print('reward 2', reward)
        # hobservation = build_states(observation)
        print(steps)
        steps += 1

if __name__ == '__main__':
    main()