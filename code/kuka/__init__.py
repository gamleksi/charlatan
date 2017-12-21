from gym.envs import register

register(
    id="KukaPoseEnv-v0",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300
)

register(
    id="KukaTrajectoryEnv-v0",
    entry_point='kuka.env:KukaTrajectoryEnv',
    timestep_limit=300,
    kwargs={'actionRepeat': 1}
)

register(
    id="KukaTrajectoryEnv-v1",
    entry_point='kuka.env:KukaTrajectoryEnv',
    timestep_limit=500,
    kwargs={'actionRepeat': 1, 'renders': False}
)

register(
    id="KukaPoseEnv-v1",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300,
    kwargs={'goalReset': False,
    'goal': [0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]}
)

register(
    id="KukaPoseEnv-v2",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=30,
    kwargs={'actionRepeat': 50}
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
    'goal': [0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]
    }
)

register (
    id="KukaTrainPoseEnv-v2",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300,
    kwargs={'goalReset': False, 'renders': False,
    'goal': [ 0.2,  0.9,  0.6,  0.1, -0.4, 0.6, -0.3, -0.8, 9.2, -2.2,  6.1,  0.4]
    }
)

register (
    id="KukaSevenJointsEnv-v0",
    entry_point='kuka.env:KukaSevenJointsEnv',
    timestep_limit=300,
    kwargs={'goalReset': False, 'renders': True,
    'goal': [0.2,  0.9,  0.6,  0.1, -0.4, 0.6, -0.3]
    }
)

register (
    id="KukaTrainSevenJointsEnv-v0",
    entry_point='kuka.env:KukaSevenJointsEnv',
    timestep_limit=300,
    kwargs={'goalReset': False, 'renders': False,
    'goal': [0.2,  0.9,  0.6,  0.1, -0.4, 0.6, -0.3]
    }
)
register (
    id="KukaTrainSevenJointsEnv-v1",
    entry_point='kuka.env:KukaSevenJointsEnv',
    timestep_limit=300,
    kwargs={'renders': False,
    }
)

register (
    id="KukaTrainSevenJointsEnv-v2",
    entry_point='kuka.env:KukaSevenJointsEnv',
    timestep_limit=300,
    kwargs={'renders': False, 'goalReset': False}
)

register (
    id="KukaTrainPoseEnv-v3",
    entry_point='kuka.env:KukaPoseEnv',
    timestep_limit=300,
    kwargs={'goalReset': False, 'renders': False,
    }
)

from imitation_util import imitation_arguments

register (
    id="KukaImitationEnv-v0",
    entry_point='imitation:ImitationEnv',
    kwargs=imitation_arguments()
)

register (
    id="ImitationWrapperEnv-v0",
    entry_point='imitation:ImitationWrapperEnv',
    kwargs={'video_dir': './data/video/angle-1', 'frame_size': (299, 299),'renders': False}       
)