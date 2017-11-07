from gym.envs import register

register(
    id="KukaPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=1000
)

register(
    id="KukaTrainPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=1000,
    kwargs={'renders' : False}
)