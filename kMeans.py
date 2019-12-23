import torch
import numpy as np
import random
import sys
import time
import os
# from sklearn.metrics import calinski_harabaz_score as CH_score
from sklearn.metrics import calinski_harabasz_score as CH_score

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')

# Choosing `num_centers` random data points as the initial centers

def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers


# Compute for each data point the closest center

def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes



# Compute new centers as means of the data points forming the clusters

def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers



def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            sys.stdout.write('\n')
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes

def cluster_eval(features):
    scores = []
    features_cpu = features.detach().cpu().numpy()
    n_clusters = list(range(50, 300, 30))
    for n_cluster in n_clusters:
        centers, labels = cluster(features, n_cluster)
        labels_cpu = labels.detach().cpu().numpy()
        score = CH_score(features_cpu, labels_cpu)
        scores.append(score)
        print("cluster num:{} cluster num:{}".format(n_cluster, score))
    scores = np.array(scores)
    return n_clusters[np.argmax(scores)]
    


if __name__ == '__main__':
    n = 30000
    d = 2048
    num_centers = 100
    # It's (much) better to use 32-bit floats, for the sake of performance
    dataset_numpy = np.random.randn(n, d).astype(np.float32)
    dataset = torch.from_numpy(dataset_numpy).to(device_gpu)
    print('Starting clustering')
    begin_time = time.time()
    centers, codes = cluster(dataset, num_centers)
    codes = codes.detach().cpu().numpy()
    print("time consume:{}".format(time.time() - begin_time))
    print(codes)
    print(codes.shape)
