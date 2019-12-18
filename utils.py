import os
from collections import defaultdict
import numbers
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import scipy.io as scio
import errno
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import time
import json
import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class pk_sampler:

    def __init__(self, cluster_result, p, k, config):

        self.config = config
        self.data_source = cluster_result
        self.cls = list(cluster_result.keys())
        self.p = p
        self.k = k


    def next_batch(self):
        '''
        数据采样
        '''
        result = []
        target = []
        cls_sampled  = np.random.choice(self.cls, size=self.p, replace=False)
        for index, _cls in enumerate(cls_sampled):
            if len(self.data_source[_cls]) < self.k:
                data_sampled = np.random.choice(self.data_source[_cls], self.k, replace=True)
            else:
                data_sampled = np.random.choice(self.data_source[_cls], self.k, replace=False)

            result.extend(data_sampled)
            target.extend([index] * self.k)
        result = [self.config.transform_train(np.array(Image.open(x).resize((224, 224), Image.ANTIALIAS))) for x in result]
        assert  len(result) == self.p * self.k
        return tuple(result), np.array(target).reshape(-1, 1)

    def __len__(self):
        return self.p * self.k


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        '''
        计算输入数据的triplets loss
        target 是每一行数据属于的label
        '''
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(tuple(dist_ap))
        dist_an = torch.stack(tuple(dist_an))
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

def cluster(net, dataset, batch_size, net_num, outIter_num, config):
    '''
    聚类函数
    '''
    avgpool = nn.AvgPool2d(7,7)
    net.eval()
    if len(dataset) % batch_size == 0:
        batch_num = len(dataset) / batch_size
    else:
        batch_num = int(len(dataset) / batch_size) + 1
    
    feat = np.zeros((len(dataset), 2048))
    image_name = []
    flag = 0
    print("==> compute feature using net{} , outIter {}".format(net_num, outIter_num))
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            images, path_list = dataset.next_batch(batch_size)
            images = Variable(torch.stack(images).cuda())
            # images = Variable(torch.Tensor(images).cuda())
            code, output = net(images)
            code = avgpool(code)
    
            assert output.shape[0] == len(path_list)
            
            tmp = code.shape[0]
            feat[flag:flag+tmp, :] = code.view(tmp,-1).detach().cpu().numpy()
            flag += tmp
            image_name.extend(path_list)
    print('features have been extracted')
    print('Begin clustering')
    # TODO 聚类函数
    n_cluster = 100
    end = time.time()
    # n_cluster = 10
    ac = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='complete')
    labels = ac.fit_predict(feat)
    print('cluster end. time spend:{}'.format(time.time() - end))
    assert len(labels) == len(image_name)
    result = {}
    for i in range(len(labels)):
        if labels[i] not in result.keys():
             result[labels[i]] = [image_name[i]]
        else:
             result[labels[i]].append(image_name[i])
#     result = {1:['data/n0441835700000007.jpg', 'data/n0441835700000015.jpg'],
#               2:['data/n0441835700000115.jpg', 'data/n0441835700000115.jpg']}

    with open(os.path.join(config.cluster_result_dir, 'cluster_result_{}_{}.json'.format(net_num, outIter_num)), 'wb') as f:
        pickle.dump(result, f)
    print('cluster result is saved to cluster_result_{}_{}.json'.format(net_num, outIter_num))
    return result
