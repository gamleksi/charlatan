import os
import argparse
import numpy as np
import torch
from PIL import Image
from util import SingleViewTripletBuilder, ensure_folder

parser = argparse.ArgumentParser()
parser.add_argument('--video-dir', type=str, default='./validation_videos')
parser.add_argument('--out', type=str, default='./validation')
parser.add_argument('--iterations', type=int, default=20)
args = parser.parse_args()

ensure_folder(args.out)
triplet_builder = SingleViewTripletBuilder('./data/validation', (128, 128), args, look_for_negative=False)
all_triplets = []
for i in range(args.iterations):
    video = triplet_builder.get_video(i % triplet_builder.video_count)
    triplets = triplet_builder.sample_triplets(video)
    all_triplets.append(triplets)

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

all_triplets = torch.cat(all_triplets, dim=0)
for index, triplet in enumerate(all_triplets):
    save_triplet(triplet, index)
