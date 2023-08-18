import datetime
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import argparse
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
import seaborn as sns
from torch.nn.parallel import DistributedDataParallel as DDP
from    copy import deepcopy

from meta import Meta
from loss import OZELoss, compute_loss
from rnnmodel import LSTM
from utils import get_bin_custom, get_bin_log, CustomBins, LogBins, get_estimate_custom
from sklearn.cluster import KMeans

import data

from cluster import cluster_data
from eval import print_metrics_binary, print_metrics_regression


parser = argparse.ArgumentParser()
parser.add_argument('--num-epoch', type=int, default=60,
                        help='number of training epochs')
parser.add_argument('--save-epoch', type=int, default=10,
                        help='frequency of model saving')
parser.add_argument('--train-way', type=int, default=6,
                        help='number of classes in one training episode')
parser.add_argument('--train-shot', type=int, default=8,
                        help='number of support examples per training class')
parser.add_argument('--train-query', type=int, default=8,
                        help='number of query examples per training class')
parser.add_argument('--test-way', type=int, default=8,
                        help='number of classes in one test (or validation) episode')
parser.add_argument('--test-shot', type=int, default=8,
                        help='number of support examples per validation class')
parser.add_argument('--test-query', type=int, default=4,
                        help='number of query examples per validation class')
parser.add_argument('--val-episode', type=int, default=2000,
                        help='number of episodes per validation')
parser.add_argument('--save-path', default='./experiments/exp_1')
parser.add_argument('--gpu', default='0, 1, 2, 3')
parser.add_argument('--network', type=str, default='ProtoNet',
                        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
parser.add_argument('--head', type=str, default='ProtoNet',
                        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
parser.add_argument('--episodes-per-batch', type=int, default=5,
                        help='number of episodes per batch')
parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')
parser.add_argument('--update-lr', type=float, default=0.005, help='epsilon of label smoothing')
parser.add_argument('--meta-lr', type=float, default=0.0005, help='epsilon of label smoothing')
parser.add_argument('--update-step', type=int, default=5, help='epsilon of label smoothing')
parser.add_argument('--update-step-test', type=int, default=5, help='epsilon of label smoothing')
parser.add_argument('--local_rank', type=int, default=2, help='epsilon of label smoothing')

def Sample(cluster_num, len_pre, len_last, epi_dom, batch_size):
    domain = random.sample(range(cluster_num), epi_dom)
    idx_unlabel_spt = [random.sample(range(len_pre[d]), batch_size) for d in domain]
    idx_unlabel_qry = [random.sample(range(len_pre[d]), batch_size) for d in domain]
    idx_label_spt = [random.sample(range(len_last[d]), batch_size) for d in domain]
    idx_label_qry = [random.sample(range(len_last[d]), batch_size) for d in domain]
    return domain, idx_unlabel_spt, idx_unlabel_qry, idx_label_spt, idx_label_qry

args = parser.parse_args()

print(datetime.datetime.now())
# Training parameters
DATASET_PATH = '/data/datasets/mimic3-benchmarks/data/decompensation'

BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 2e-3
EPOCHS = 3000

# Model parameters
d_input = 76 # From dataset
d_output = 1 # From dataset

config = [d_input, 128, d_output, dropout]

# Config
sns.set()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

cluster_num = 5
cluster_epi = 3
data_train, aug_train = data.load_all_data(DATASET_PATH,BATCH_SIZE,100,mode='train')
data_test, aug_test = data.load_all_data(DATASET_PATH,BATCH_SIZE,100,mode='val')

cluster_pre_train, cluster_aug_train, cluster_last_train, cluster_y_train, cluster_yt_train, clusters_train = cluster_data(data_train, aug_train, cluster_num)
cluster_pre_test, cluster_aug_test, cluster_last_test, cluster_y_test, cluster_yt_test, clusters_test = cluster_data(data_test, aug_train, cluster_num, aug_num=3000)

model_save_path = '/home/grads/l/lidazhang/dec/maml_semi_15_tau15'
SAVED_PATH = '/home/grads/l/lidazhang/dec/onelayer128'
net_load = LSTM(d_input, 128, d_output, dropout, BATCH_SIZE).to(device)
net_load.load_state_dict(torch.load(SAVED_PATH))
param_list = list(net_load.parameters())

maml = Meta(args, config, device).to(device)
# maml = nn.DataParallel(maml)
for i, p in enumerate(maml.parameters()):
    try:
        p.data = param_list[i]
    except:
        break

val_loss_best = np.inf
best_auc = 0

# Prepare loss history
hist_loss = np.zeros(EPOCHS)
hist_loss_val = np.zeros(EPOCHS)
opt = optim.Adam(maml.parameters(), lr=args.meta_lr)

len_pre = [len(cluster_pre_train[i]) for i in cluster_pre_train.keys()]
len_last = [len(cluster_last_train[i]) for i in cluster_last_train.keys()]
print(len_pre, len_last)

for idx_epoch in range(EPOCHS):
    print(datetime.datetime.now())
    running_loss = 0
    train_loss = []
    #with tqdm(total=len(data_train), desc=f"[Epoch {idx_epoch+1:3d}/{EPOCHS}]") as pbar:
    #for idx_batch, batch in enumerate(tqdm(dloader_train(idx_epoch)), 1):
    for idx_batch in range(args.episodes_per_batch):
        print(idx_batch)

        domain, idx_unlabel_spt, idx_unlabel_qry, idx_label_spt, idx_label_qry = Sample(cluster_num, len_pre, len_last, cluster_epi, BATCH_SIZE)

        x_spt = [torch.Tensor(cluster_last_train[domain[i]][idx_label_spt[i]]).cuda() for i in range(len(domain))]
        y_spt = [torch.Tensor(cluster_y_train[domain[i]][idx_label_spt[i]]).cuda() for i in range(len(domain))]
        x_qry = [torch.Tensor(cluster_last_train[domain[i]][idx_label_spt[i]]).cuda() for i in range(len(domain))]
        y_qry = [torch.Tensor(cluster_y_train[domain[i]][idx_label_spt[i]]).cuda() for i in range(len(domain))]

        xu_spt = [torch.Tensor(cluster_pre_train[domain[i]][idx_unlabel_spt[i]]).cuda() for i in range(len(domain))]
        xaug_spt = [torch.Tensor(cluster_aug_train[domain[i]][idx_unlabel_spt[i]]).cuda() for i in range(len(domain))]
        xu_qry = [torch.Tensor(cluster_pre_train[domain[i]][idx_unlabel_qry[i]]).cuda() for i in range(len(domain))]
        xaug_qry = [torch.Tensor(cluster_aug_train[domain[i]][idx_unlabel_qry[i]]).cuda() for i in range(len(domain))]

        torch.cuda.empty_cache()
        meta_loss = maml((x_spt, y_spt, x_qry, y_qry, xu_spt, xaug_spt, xu_qry, xaug_qry))
        opt.zero_grad()
        meta_loss.mean().backward()
        torch.nn.utils.clip_grad_value_(maml.parameters(), clip_value = 10.0)
        opt.step()
        train_loss.append(meta_loss.cpu().detach().numpy())

    if idx_epoch % 1 == 0:
        print('idx_epoch:', idx_epoch)

    for p in maml.parameters():
        print(p)
        break

    roc_all = []
    prc_all = []
    for w in range(len(cluster_pre_test)):
        l = list(cluster_pre_test.keys())[w]
        print('w',w,l)

        # x_spt = torch.Tensor(train_data[l][:BATCH_SIZE]).cuda()
        # y_spt = torch.Tensor(train_label[l][:BATCH_SIZE]).cuda()
        x_spt = torch.Tensor(cluster_last_test[l][:BATCH_SIZE]).cuda()
        y_spt = torch.Tensor(cluster_y_test[l][:BATCH_SIZE]).cuda()
        x_qry = torch.Tensor(cluster_last_test[l][:BATCH_SIZE]).cuda()
        y_qry = torch.Tensor(cluster_y_test[l][:BATCH_SIZE]).cuda()

        fast_weights = maml((x_spt, y_spt, x_qry, y_qry, xu_spt, xaug_spt, xu_qry, xaug_qry), meta_train=False, fast_weights=None)

        y_pred_all = []
        y_true_all = []
        for i in range(1, len(cluster_last_test[l])//BATCH_SIZE):
            x_qry = torch.Tensor(cluster_last_test[l][i*BATCH_SIZE: (i+1)*BATCH_SIZE]).cuda()
            yt_qry = cluster_yt_test[l][i*BATCH_SIZE: (i+1)*BATCH_SIZE]

            netout = maml((None, None, x_qry, y_qry, None, None, None, None), meta_train=False, fast_weights=fast_weights)
            # output = net_load(x_qry)
            y_pred_all.extend(netout.cpu().detach().numpy())
            y_true_all.extend(yt_qry)

        # y_true = np.reshape(y_true_all, (-1,1))
        # y_pred = np.reshape(y_pred_all, (-1,1))
        predictions = y_pred_all
        roc, prc = print_metrics_binary(y_true_all, predictions)
        roc_all.append(roc)
        prc_all.append(prc)
    mean_roc = np.mean(roc_all)
    mean_prc = np.mean(prc_all)
    mean_auc = mean_roc + mean_prc
    print('Average ROC:', mean_roc)
    print('Average PRC:', mean_prc)
    print('Average AUC:', mean_auc)
    if mean_auc > best_auc:
        best_auc = mean_auc
        # torch.save(maml.state_dict(), model_save_path)
    print('Best AUC:', best_auc)
