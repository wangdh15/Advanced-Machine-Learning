import os
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Sampler

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
#         return img
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



class  myDataset(data.Dataset):
    '''
    定义选取整个数据集的dataset，用于计算feature然后聚类
    '''
    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config = config
        self.transform = config.transform_train

    def __getitem__(self, index):

        img_path = self.file_path[index]
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = self.transform(np.array(img))
        return img, img_path

    def __len__(self):
        return len(self.file_path)


class myDataset2(data.Dataset):
    '''
    聚类结果对应的dataset
    需要和自己写的Sampler配合使用
    '''
    def __init__(self, cluster_result, config):
        '''
        本质上存这些东西没什么用
        '''
        self.cluster_result = cluster_result
        self.config = config
        self.transform = config.transform_train
        self.length = 0
        for x in self.cluster_result.values():
            self.length += len(x)

    def __getitem__(self, item):
        '''
        这里的index是从自己写的sampler中拿到的路径
        自己定义的sampler中没有返回index，而是直接返回文件路劲
        '''
        img = Image.open(item)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = self.transform(np.array(img))
        label = None
        for key in self.cluster_result.keys():
            if item in self.cluster_result[key]:
                label = key
        return img,label

    def __len__(self):
        return self.length


class pkSampler(Sampler):
    '''
    pk采样器的实现，配合上述的聚类结果对应的dataset进行使用
    替换默认的batch_sampler
    '''
    def __init__(self, cluster_result, config):
        super(pkSampler, self).__init__(cluster_result)
        self.data_source = cluster_result
        self.cls = list(cluster_result.keys())
        self.config = config
        self.p = config.p
        self.k = config.k
        self.length = 0
        for x in self.data_source.values():
            self.length += len(x)

    def __iter__(self):
        '''
        不断返回采样得到的数据
        返回的是这个batch使用的数据的路径组成的list
        之后由dataset中的getitem遍历，作为函数__getitem__的参数，得到最终的一个batch的数据
        '''
        while True:
            batch = []
            cls_sampled  = np.random.choice(self.cls, size=self.p, replace=False)
            for _cls in enumerate(cls_sampled):
                if len(self.data_source[_cls]) < self.k:
                    data_sampled = np.random.choice(self.data_source[_cls], self.k, replace=True)
                else:
                    data_sampled = np.random.choice(self.data_source[_cls], self.k, replace=False)

                batch.extend(data_sampled)
            assert  len(batch) == self.p * self.k
            yield batch

    def __len__(self):
        return self.length

