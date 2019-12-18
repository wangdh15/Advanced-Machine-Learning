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
# from dataloader import Imagenet_data
from dataloader import data_loader


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


def adjust_lambda(epoch):
    '''
    adjust the lambda
    '''
    if epoch < 50:
        lambda_1 = 0.9
        lambda_2 = 0.1
    elif epoch >= 50 and epoch < 80:
        lambda_1 = 0.5
        lambda_2 = 0.5
    else:
        lambda_1 = 0.1
        lambda_2 = 0.9
    return lambda_1, lambda_2


# def train_each_epoch(epoch, net, trainloader, optimizer, criterion):
#     current_lr = adjust_learning_rate(optimizer, epoch)
#     train_loss = AverageMeter()
#     data_time = AverageMeter()
#     batch_time = AverageMeter()
#
#     # switch to train mode
#     net.train()
#     end = time.time()
#     for batch_idx, images in enumerate(trainloader):
#
#         images = Variable(images.cuda())
#         # images = Variable(images)
#         data_time.update(time.time() - end)
#         outputs = net(images)
#         loss = criterion(images, outputs)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss.update(loss.item(), images.size(0))
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#         if batch_idx % 10 == 0:
#             print('Epoch: [{}][{}/{}] '
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
#                   'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
#                   'lr:{} '
#                   'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
#                   .format(epoch, batch_idx, len(trainloader), current_lr, batch_time=batch_time,
#                           data_time=data_time, train_loss=train_loss))
#
#
# def train(config):
#
#
#     np.random.seed(config.np_random_seed)
#     data_dir = config.data_dir
#     dataset = config.dataset
#     log_path = config.log_path
#     checkpoint_path = config.model_path
#     suffix = dataset
#     sys.stdout = Logger(log_path + suffix + '_os.txt')
#     start_epoch = 1
#     print('==> Loading data..')
#     end = time.time()
#     Img_dataset = Imagenet_data(config)
#
#     print('Dataset {} statistics:'.format(dataset))
#     print('  ------------------------------')
#     print('  Images  |  {:8d}'.format(len(Img_dataset)))
#     print('  ------------------------------')
#     print('Data Loading Time:\t {:.3f}'.format(time.time() - end))
#
#     print('==> Building model..')
#
#     net = encoder_decoder()
#     if config.gpu:
#         net = nn.DataParallel(net).cuda()
#     optimizer = optim.SGD([{'params': net.parameters(), 'lr': config.lr}], weight_decay=5e-4, momentum=0.9,
#                           nesterov=True)
#     # load model
#     if len(config.resume) > 0:
#         model_path = checkpoint_path + config.resume
#         if os.path.isfile(model_path):
#             print('==> loading checkpoint {}'.format(config.resume))
#             checkpoint = torch.load(model_path)
#             start_epoch = checkpoint['epoch']
#             net.load_state_dict(checkpoint['net'])
#             print('==> loaded checkpoint {} (epoch {})'
#                   .format(config.resume, checkpoint['epoch']))
#         else:
#             print('==> no checkpoint found at {}'.format(config.resume))
#
#     criterion = nn.MSELoss()
#
#     # training
#     print('==> Start Training...')
#     for epoch in range(start_epoch, 101):
#
#         trainloader = data.DataLoader(Img_dataset, batch_size=config.batch_size, num_workers=config.workers,
#                                       drop_last=True)
#
#         train_each_epoch(epoch, net, trainloader, optimizer ,criterion)
#
#         if epoch > 10 and epoch % config.save_epoch == 0:
#             print('Save model Epoch: {}'.format(epoch))
#             state = {
#                 'net': net.state_dict(),
#                 'epoch': epoch,
#             }
#             torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
def train_signal_model(net, cluster_result, config, optimizer, _iter, net_num):
    '''
    训练单个模型的过程
    '''
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    sampler = pk_sampler(cluster_result, config.p, config.k, config)
    criterion1 = nn.MSELoss()
    criterion2 = TripletLoss()

    net.train()

    for batchid in range(config.iter_each_net):


        end = time.time()
        data, target = sampler.next_batch()
        data_time.update(time.time() - end)

        data = Variable(torch.stack(data).cuda())

        target = Variable(torch.Tensor(target).cuda())

        code, output = net(data)

        code = code.view(code.shape[0], -1)
        # 根据迭代次数调整loss的比例以及学习率
        lambda_1, lambda_2 = adjust_lambda(batchid)

        loss1 = criterion1(data, output)
        loss2, _ = criterion2(code, target)
        loss = lambda_1 * loss1 +lambda_2 * loss2
        current_lr = adjust_learning_rate(optimizer, batchid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batchid % 10 == 0:
            print('net: {} '
                  'Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'lr:{} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  .format( net_num, _iter, batchid, config.iter_each_net , current_lr, batch_time=batch_time,
                          data_time=data_time, train_loss=train_loss))

        # 一定的batch_id存储模型
        if batchid > 0 and batchid % 100 == 0:
            print('Save model OutIter: {}  batchId : {}'.format( _iter,batchid))
            state = {
                'net': net.state_dict(),
                'outIter' : _iter,
                'batchid': batchid,
            }
            torch.save(state, config.model_path + '_net_{}' + '_batchid_{}.t'.format(net_num,batchid))


def train(config):
    '''
    两个encoder_decoder迭代训练
    '''
    # load data
    np.random.seed(config.np_random_seed)
    data_dir = config.data_dir
    dataset = config.dataset
    log_path = config.log_path
    checkpoint_path = config.model_path
    suffix = dataset
    sys.stdout = Logger(log_path + suffix + '_os.txt')
    outIter = 0
    print('==> Loading data..')
    end = time.time()

    file_list = os.listdir(config.data_dir)
    file_list = [ os.path.join(config.data_dir, x) for x in file_list]
    file_list_1 = np.random.choice(file_list, int(len(file_list)/2), replace=False)
    file_list_2 = [ x for x in file_list if x  not in file_list_1]
    dataset_all = data_loader(file_list, config)
    dataset_1 = data_loader(file_list_1, config)
    dataset_2 = data_loader(file_list_2, config)

    print('Dataset {} statistics:'.format(dataset))
    print('  ------------------------------')
    print('  Images_all  |  {:8d}'.format(len(dataset_all)))
    print('  Images_1  |  {:8d}'.format(len(dataset_1)))
    print('  Images_2  |  {:8d}'.format(len(dataset_2)))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    # define model
    net1 = encoder_decoder()
    net2 = encoder_decoder()
    if config.gpu:
        net1 = nn.DataParallel(net1).cuda()
        net2 = nn.DataParallel(net2).cuda()

    # load model from pretrain model
    if len(config.resume_net1) > 0:
        model_path = checkpoint_path + config.resume_net1
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(config.resume_net1))
            checkpoint = torch.load(model_path)
            # outIter = checkpoint['outIter']
            net1.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(config.resume_net1, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(config.resume_net1))

    if len(config.resume_net2) > 0:
        model_path = checkpoint_path + config.resume_net2
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(config.resume_net2))
            checkpoint = torch.load(model_path)
            # outIter = checkpoint['outIter']
            net2.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(config.resume_net2, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(config.resume_net2))

    # Loss function
    criterion1 = nn.MSELoss()
    criterion2 = TripletLoss()
    optimizer1 = optim.SGD([{'params': net1.parameters(), 'lr': config.lr}], weight_decay=5e-4, momentum=0.9,
                          nesterov=True)
    optimizer2 = optim.SGD([{'params': net2.parameters(), 'lr': config.lr}], weight_decay=5e-4, momentum=0.9,
                          nesterov=True)

    # training
    print('==> Start Training...')
    for _iter in range(outIter, config.iter_num):
        cluster_result = cluster(net1, dataset_2 , 64)
        train_signal_model(net2, cluster_result, config, optimizer2, _iter, 2)
        cluster_result = cluster(net2, dataset_1, 64)
        train_signal_model(net1, cluster_result, config, optimizer1, _iter, 1)


if __name__ == '__main__':
    train(config)
