import os
import argparse
import numpy as np
import torch
from PIL import Image
from util import SingleViewTripletBuilder, ensure_folder
from torch.utils.data import ConcatDataset

parser = argparse.ArgumentParser()
parser.add_argument('--video-dir', type=str, default='./validation_videos')
parser.add_argument('--out', type=str, default='./validation')
parser.add_argument('--iterations', type=int, default=2)
args = parser.parse_args()

ensure_folder(args.out)
triplet_builder = SingleViewTripletBuilder('./data/validation', (299, 299), args, sample_size=100)
datasets = []
for i in range(args.iterations):
    dataset = triplet_builder.build_set()
    datasets.append(dataset)

def to_image(triplet):
    np_array = triplet.numpy()
    return Image.fromarray(
        np.transpose(np_array * 255, [1, 2, 0]).astype(np.uint8), 'RGB')

def save_triplet(triplet, index):
    anchor_frame = triplet[0, :, :, :]
    positive_frame = triplet[1, :, :, :]
    negative_frame = triplet[2, :, :, :]
    to_image(anchor_frame).save(os.path.join(args.out, '{0}-a.jpg'.format(index)), 'JPEG')
    to_image(positive_frame).save(os.path.join(args.out, '{0}-p.jpg'.format(index)), 'JPEG')
    to_image(negative_frame).save(os.path.join(args.out, '{0}-n.jpg'.format(index)), 'JPEG')
    print('generated images for ', index)

for index, triplet in enumerate(ConcatDataset(datasets)):
    triplet = triplet[0]
    save_triplet(triplet, index)
