import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

# class Imagenet_data(data.Dataset):
#     def __init__(self, l_path, config):
#         # Load training images (path) and labels
#         # self.train_image = np.load(data_dir + 'train_img.npy')
#         self.config = config
#         self.transform = config.transform_train
#         self.img_paths = l_path
#
#
#     def _image_path(self):
#         '''
#         对数据集进行随机划分为两份
#         '''
#         tmp = os.listdir(self.config.data_dir)
#         np.random.seed(self.config.np_random_seed)
#         assert len(tmp) == 60000
#         np.random.shuffle(tmp)
#         result_tmp = [os.path.join(self.config.data_dir, x) for x in tmp]
#         return result_tmp[0:30000], result_tmp[30000:60000]
#
#     def __getitem__(self, index):
#         img_path = self.img_paths[index]
#         img = Image.open(img_path)
#         img = img.resize((224, 224), Image.ANTIALIAS)
#         img = self.transform(np.array(img))
#         return
#
#     def __len__(self):
#         return len(self.train_image_1) + len(self.train_image_2)


# class data_loader:
#
#     def __init__(self, config):
#         # Load training images (path) and labels
#         # self.train_image = np.load(data_dir + 'train_img.npy')
#         self.config = config
#         self.train_image_1, self.train_image_2, self.train_image_all = self._image_path()
#         self.transform = config.transform_train
#         self.start_batch_index_1 = 0
#         self.start_batch_index_2 = 0
#         self.start_batch_index_all = 0
#
#
#     def train_1_len(self):
#         return len(self.train_image_1)
#
#     def train_2_len(self):
#         return len(self.train_image_2)
#
#     def _image_path(self):
#         '''
#         对数据集进行随机划分为两份
#         '''
#         tmp = os.listdir(self.config.data_dir)
#         np.random.seed(self.config.np_random_seed)
#         # assert len(tmp) == 60000
#         np.random.shuffle(tmp)
#         result_tmp = [os.path.join(self.config.data_dir, x) for x in tmp]
#         return result_tmp[0:len(result_tmp)/2], result_tmp[len(result_tmp)/2:-1], result_tmp
#
#     def next_batch_1(self, batch_size):
#         '''
#         获取第一个模型的训练数据
#         '''
#         if self.start_batch_index_1 + batch_size >= len(self.train_image_1):
#             path_list = self.train_image_1[
#                         self.start_batch_index_1:-1]
#             self.start_batch_index_1 = 0
#         else:
#             path_list = self.train_image_1[self.start_batch_index_1:self.start_batch_index_1 + batch_size]
#             self.start_batch_index_1 += batch_size
#
#         return [self.transform(np.array(Image.open(x).resize((224, 224), Image.ANTIALIAS))) for x in path_list], path_list
#
#     def next_batch_2(self, batch_size):
#         '''
#         或许第二个模型的训练数据
#         '''
#         if self.start_batch_index_2 + batch_size >= len(self.train_image_2):
#             path_list = self.train_image_2[
#                         self.start_batch_index_2:-1]
#             self.start_batch_index_2 = 0
#         else:
#             path_list = self.train_image_2[self.start_batch_index_2:self.start_batch_index_2 + batch_size]
#             self.start_batch_index_2 += batch_size
#
#         return [self.transform(np.array(Image.open(x).resize((224, 224), Image.ANTIALIAS))) for x in path_list], path_list
#
#     def next_batch_all(self, batch_size):
#         '''
#         获取全部的训练数据
#         '''
#         if self.start_batch_index_all + batch_size >= len(self.train_image_all):
#             path_list = self.train_image_all[
#                         self.start_batch_index_all:-1]
#             self.start_batch_index_all = 0
#         else:
#             path_list = self.train_image_all[self.start_batch_index_all:self.start_batch_index_all + batch_size]
#             self.start_batch_index_all += batch_size
#
#         return [self.transform(np.array(Image.open(x).resize((224, 224), Image.ANTIALIAS))) for x in path_list], path_list
#
#     def __len__(self):
#         return len(self.train_image_all)


class data_loader:

    def __init__(self, file_list, config):
        self.file_list = file_list
        self.batch_start_index = 0
        self.transform = config.transform_train

    def __len__(self):
        return len(self.file_list)

    def reset(self):
        self.batch_start_index = 0

    def next_batch(self, batch_size):
        if self.batch_start_index + batch_size >= len(self.file_list):
            path_list = self.file_list[self.batch_start_index:]
            self.batch_start_index = 0
        else:
            path_list = self.file_list[self.batch_start_index:self.batch_start_index+batch_size]
            self.batch_start_index += batch_size

        return [self.transform(np.array(Image.open(x).resize((224, 224), Image.ANTIALIAS))) for x in path_list], path_list
