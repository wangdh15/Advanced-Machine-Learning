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
data_dir = "data/"

# ----------------------------
# config for trainer
lr = 0.01   # learning rate
model_path = "save_model/"   # model save path
if not os.path.isdir(model_path):
    os.makedirs(model_path)
resume_net1 = 'miniImagenet_epoch_80.t'   # resume from checkpoint
if not os.path.exists(os.path.join(model_path, resume_net1)):
    resume_net1 = ""
resume_net2 = 'miniImagenet_epoch_80.t'   # resume from checkpoint
if not os.path.exists(os.path.join(model_path, resume_net2)):
    resume_net2 = ""


save_epoch = 10    # save model every 10 epoch
log_path = "log/"   # log save path
if not os.path.isdir(log_path):
    os.makedirs(log_path)

batch_size = 32  # training batch size
workers = 4  # number of data loading workers
np_random_seed = 0   # numpy random seed
iter_num = 10
gpu='2'  # visible gpu
iter_each_net = 200
lambda_1 = 0.5
lambda_2 = 0.5

# ----------------------------
# config for pk sample

p = 2
k = 1