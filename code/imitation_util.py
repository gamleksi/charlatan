import os
from util import normalize
from tcn import define_model
import torch

def load_model(use_cuda):
    import torch
    tcn = define_model(use_cuda)
    model_path = os.path.join(
        "./trained_models/tcn",
        "inception-epoch-2000.pk"
    )
    tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    return tcn

def imitation_arguments(renders=False):
    # by default cuda is used if it is available
    args = {}
    args['video_dir'] = './data/validation/angle1'
    args['frame_size'] = (299, 299)
    args['renders'] = renders
    return args