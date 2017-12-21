import gym
from imitation import ImitationWrapperEnv, TCNWrapperEnv
import numpy as np

def main():
    tcn = TCNWrapperEnv()
    env = gym.make('ImitationWrapperEnv-v0')
    env.seed(2222)
    done = False
    steps = 0

    while not(done):
        observation, reward, done, _ = env.step(env.action_space.sample())
        video_frames = reward[0] 
        current_frames = reward[1]
        reward = tcn.reward2(np.array([video_frames]), np.array([current_frames]))
        print('reward 1', reward)
        reward = tcn.reward(np.array([video_frames]), np.array([current_frames]))
        print('reward 2', reward)
        # hobservation = build_states(observation)
        print(steps)
        steps += 1
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()