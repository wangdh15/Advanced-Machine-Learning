from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from model import encoder_decoder
from utils import *
from dataloader import Imagenet_data

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='miniImagenet', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=10, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--img_w', default=224, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=224, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
np.random.seed(0)
# data_dir = '/data/douzp/miniImagenet/images/'
data_dir = '/home/hanj/self-supervised/miniImagenet/images/'
dataset = args.dataset
log_path = args.log_path
checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

suffix = dataset
test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
end = time.time()
Img_dataset = Imagenet_data(data_dir, transform=transform_train)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  Images  |  {:8d}'.format(len(Img_dataset)))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = encoder_decoder()
optimizer = optim.SGD([{'params': net.parameters(), 'lr': args.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
net = nn.DataParallel(net).cuda()

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

criterion = nn.MSELoss()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 50:
        lr = args.lr
    elif epoch >= 50 and epoch < 80:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.05
    optimizer.param_groups[0]['lr'] = lr
    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, images in enumerate(trainloader):
        images = Variable(images.cuda())
        data_time.update(time.time() - end)
        outputs = net(images)
        loss = criterion(images, outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  .format(epoch, batch_idx, len(trainloader), current_lr, batch_time=batch_time,
                          data_time=data_time, train_loss=train_loss))


# training
print('==> Start Training...')
for epoch in range(start_epoch, 101 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    trainloader = data.DataLoader(Img_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 10 and epoch % args.save_epoch == 0:
        print('Save model Epoch: {}'.format(epoch))
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
