import os
import numpy as np
import torch.utils.data as data
from PIL import Image


class Imagenet_data(data.Dataset):
    def __init__(self, data_dir, transform=None):
        # Load training images (path) and labels
        # self.train_image = np.load(data_dir + 'train_img.npy')
        self.train_image = self._image_path(data_dir)
        self.transform = transform

    def _image_path(self, data_dir):
        tmp = os.listdir(data_dir)
        return [os.path.join(data_dir, x) for x in tmp]

    def __getitem__(self, index):
        img_path = self.train_image[index]
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = self.transform(np.array(img))
        return img

    def __len__(self):
        return len(self.train_image)
