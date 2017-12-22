import os

def imitation_arguments(renders=False):
    # by default cuda is used if it is available
    args = {}
    args['video_dir'] = './data/validation/angle1'
    args['frame_size'] = (299, 299)
    args['renders'] = renders
    return args