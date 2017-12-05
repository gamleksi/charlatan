import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from util import ls

class ImageSampler(Dataset):
    # Sampler for saved validation images.
    # The directory structure is such that images in the folder are such
    # that the anchor image filename is postfixed with -a.jpg, positive with -p.jpg and
    # negative with -n.jpg
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_count = self._count_images()

    def __len__(self):
        return self.image_count

    def __get__item(self, i):
        anchor_image = Image.open(self._filepath('{0}-a.jpg'.format(i)))
        positive_image = Image.open(self._filepath('{0}-p.jpg'.format(i)))
        negative_image = Image.open(self._filepath('{0}-n.jpg'.format(i)))
        tensors = [self._to_tensor(anchor_image),
            self._to_tensor(positive_image),
            self._to_tensor(negative_image)]
        return torch.stack(tensors, dim=0)

    def _to_tensor(self, image):
        np_array = np.array(image)
        return torch.FloatTensor(np_array)

    def _filepath(self, name):
        return os.path.join(self.image_dir, name)

    def _count_images(self):
        filenames = ls(self.image_dir)
        return len(filenames) // 3


