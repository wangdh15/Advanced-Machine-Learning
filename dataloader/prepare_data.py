import os
import numpy as np
from PIL import Image

data_path = '/data/douzp/miniImagenet/images/'


def read_imgs(path):
    train_img = []
    train_image = os.listdir(path)
    for i in train_image:
        img_path = os.path.join(path, i)
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        pix_array = np.array(img)
        train_img.append(pix_array)
    return np.array(train_img)


train_img = read_imgs(data_path)
np.save(data_path + 'train_img.npy', train_img)
