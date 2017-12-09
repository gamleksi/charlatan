from gym.envs import register

register(
    id="KukaPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300
)

register(
    id="KukaPoseEnv-v1",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300,
    kwargs={'goalReset': False,
    'goal': [0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0]}
)

register (
    id="KukaTrainPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300,
    kwargs={'renders' : False}
)

register (
    id="KukaTrainPoseEnv-v1",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300,
    kwargs={'goalReset': False, 'renders': False,
    'goal': [0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
)


# from imitation_util import imitation_arguments 

# register (
#    id="KukaImitationEnv-v0",
#    entry_point='imitation:ImitationEnv',
#    kwargs=imitation_arguments()
#)