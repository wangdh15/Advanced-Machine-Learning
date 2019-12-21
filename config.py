import torchvision.transforms as transforms
import os

# ----------------------------
# config for dataloader
dataset = 'miniImagenet'      # dataset name
# transform_train
img_w = 224   # img width
img_h = 224   # img height
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((img_h, img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
# data_dir = "data/"
data_dir =  '/home/tsinghuaee09/01.Datasets_AML/images'
# data_dir =  '/home/tsinghuaee09/01.Datasets_AML/debug_dataset'
# data_dir = 'data/debug_dataset'

# ----------------------------
# config for trainer
lr = 0.001   # learning rate
model_path = "save_model/"   # model save path
if not os.path.isdir(model_path):
    os.makedirs(model_path)
resume_net1 = 'miniImagenet_epoch_100.t'   # resume from checkpoint
if not os.path.exists(os.path.join(model_path, resume_net1)):
    resume_net1 = ""
resume_net2 = 'miniImagenet_epoch_100.t'   # resume from checkpoint
if not os.path.exists(os.path.join(model_path, resume_net2)):
    resume_net2 = ""

margin = 1
save_epoch = 10    # save model every 10 epoch
log_dir = "log/"   # log save path
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
log_path = log_dir + "n_cluster_300.txt"


epoch = 10
# batch_size = 32  # training batch size
workers = 4  # number of data loading workers
np_random_seed = 0   # numpy random seed
iter_num = 10
gpu='0,1'  # visible gpu
# lambda_1 = 0.5
# lambda_2 = 0.5
cluster_batch_size = 256
cluster_result_dir = "cluster_result/"
if not os.path.isdir(cluster_result_dir):
    os.makedirs(cluster_result_dir)

# debug setting
# n_cluster = 10
# run setting
n_cluster = 300
# ----------------------------
# config for pk sample
# 单张卡大概可以跑80张图（16G显存）
# run setting
p = 16
k = 8

# debug setting
# p = 2
# k = 5
