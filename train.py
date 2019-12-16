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

import config
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 50:
        lr = config.lr
    elif epoch >= 50 and epoch < 80:
        lr = config.lr * 0.1
    else:
        lr = config.lr * 0.05
    optimizer.param_groups[0]['lr'] = lr
    return lr


def train_each_epoch(epoch, net, trainloader, optimizer, criterion):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    net.train()
    end = time.time()
    for batch_idx, images in enumerate(trainloader):
        images = Variable(images.cuda())
        # images = Variable(images)
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


def train(config):


    np.random.seed(config.np_random_seed)
    data_dir = config.data_dir
    dataset = config.dataset
    log_path = config.log_path
    checkpoint_path = config.model_path
    suffix = dataset
    sys.stdout = Logger(log_path + suffix + '_os.txt')
    start_epoch = 1
    print('==> Loading data..')
    end = time.time()
    Img_dataset = Imagenet_data(config)

    print('Dataset {} statistics:'.format(dataset))
    print('  ------------------------------')
    print('  Images  |  {:8d}'.format(len(Img_dataset)))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    print('==> Building model..')

    net = encoder_decoder()
    if config.gpu:
        net = nn.DataParallel(net).cuda()
    optimizer = optim.SGD([{'params': net.parameters(), 'lr': config.lr}], weight_decay=5e-4, momentum=0.9,
                          nesterov=True)
    # load model
    if len(config.resume) > 0:
        model_path = checkpoint_path + config.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(config.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(config.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(config.resume))

    criterion = nn.MSELoss()

    # training
    print('==> Start Training...')
    for epoch in range(start_epoch, 101):

        trainloader = data.DataLoader(Img_dataset, batch_size=config.batch_size, num_workers=config.workers,
                                      drop_last=True)

        train_each_epoch(epoch, net, trainloader, optimizer ,criterion)

        if epoch > 10 and epoch % config.save_epoch == 0:
            print('Save model Epoch: {}'.format(epoch))
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

if __name__ == '__main__':
    train(config)
