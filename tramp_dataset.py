#code adopted from Paritosh Parmar (https://github.com/ParitoshParmar/C3D-LSTM--PyTorch)

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from opts import *

torch.manual_seed(random_seed);
torch.cuda.manual_seed_all(random_seed);
random.seed(random_seed);
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True


def load_image(image_path, hori_flip, transform):
    image = Image.open(image_path)
    if config['feat_extractor'] == "c3d" or config['feat_extractor'] == "r21d":
        size = input_resize
        interpolator_idx = random.randint(0, 3)
        interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        interpolator = interpolators[interpolator_idx]
        image = image.resize(size, interpolator)
    if hori_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


class TrampDataset(Dataset):

    def __init__(self, annotations, mode, vid_mean=[0, 0, 0], vid_std=[1, 1, 1], score_mean=0, score_std=1):
        super(TrampDataset, self).__init__()
        self.annotations = annotations
        self.mode = mode

        self.vid_mean = vid_mean
        self.vid_std = vid_std
        self.score_mean = score_mean
        self.score_std = score_std

        self.annotations = self.annotations[:10]

    def __getitem__(self, ix):
        sample_no = int(self.annotations[ix][0])

        image_list = sorted((glob.glob(os.path.join(dataset_dir, 'frames',
                                                    str('{:03d}'.format(sample_no)), '*.jpg'))))

        pre_transform = transforms.Compose([transforms.CenterCrop(H),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.vid_mean, self.vid_std)])

        images = torch.zeros(sample_length, C, H, W)
        if self.mode == "train":
            hori_flip = bool(random.getrandbits(1))
        else:
            hori_flip = False

        for i in np.arange(sample_length):
            images[i] = load_image(image_list[i], hori_flip, pre_transform)

        if self.mode == "train" and config['transforms'] is not None:
            images = config['transforms'](images)

        label_final_score = (self.annotations[ix][1] - self.score_mean) / self.score_std
        data = {'video': images, 'label_final_score': label_final_score}

        return data

    def __len__(self):
        return len(self.annotations)