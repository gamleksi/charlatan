from gym.envs import register

register(
    id="KukaPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=1000
)

register(
    id="KukaPoseEnv-v1",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=1000,
    kwargs={'goalReset': False, 
    'goal': [0.28966519,  0.90138522,  0.60981021,0.18800668, -0.45304158,  0.61112126, -0.38125851, -0.891773, 9.26835694,-0.38344152,  6.05119078,  0.78474531, -0.56804456,  8.58217142]
    }
)

register(
    id="KukaTrainPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=1000,
    kwargs={'renders' : False}
)

register (
    id="KukaTrainPoseEnv-v1",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=1000,
    kwargs={'goalReset': False, 'renders': False, 
    'goal': [0.28966519,  0.90138522,  0.60981021,0.18800668, -0.45304158,  0.61112126, -0.38125851, -0.891773, 9.26835694,-0.38344152,  6.05119078,  0.78474531, -0.56804456,  8.58217142]
    }
)