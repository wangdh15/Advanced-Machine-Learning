import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class Imagenet_data(data.Dataset):
    def __init__(self, config):
        # Load training images (path) and labels
        # self.train_image = np.load(data_dir + 'train_img.npy')
        self.config = config
        self.train_image = self._image_path()
        self.transform = config.transform_train


    def _image_path(self):
        tmp = os.listdir(self.config.data_dir)
        return [os.path.join(self.config.data_dir, x) for x in tmp]

    def __getitem__(self, index):
        img_path = self.train_image[index]
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = self.transform(np.array(img))
        return img

    def __len__(self):
        return len(self.train_image)
