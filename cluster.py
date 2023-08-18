import os
import numpy as np
from sklearn import metrics
import argparse
import random

from sklearn.cluster import KMeans


def cluster_data(data_val, aug_val, n_cluster=10, aug_num=-1):
    x_all = []
    x_pre = []
    x_last = []
    y_all = []
    yt_all = []
    # for (x, y) in data_train: #xx: without normalizer
    #     x_all.extend(list(x))
    #     y_all.extend(y)
    for (x, y, yt) in data_val: #xx: without normalizer
        x_pre.extend(list(x[:-1]))
        x_last.append(x[-1])
        y_all.append(y)
        yt_all.append(yt)

    if aug_num > 0:
        x_pre = x_pre[:aug_num]
    # x_all = np.reshape(x_all,(-1,24*76))
    print(np.shape(x_last))
    print(np.shape(x_pre))
    x_all = np.concatenate((x_last, x_pre), axis=0)
    x_all = np.reshape(x_all, (-1, 24, 76))
    x_cluster = np.reshape(x_all[:,:,-17:],(-1,24*17))

    length = list(range(n_cluster))
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(x_cluster)
    # print(list(kmeans.labels_))
    print(n_cluster,kmeans.inertia_)
    cluster_pre, cluster_aug, cluster_last, cluster_y, cluster_yt = {}, {}, {}, {}, {}
    for i in range(n_cluster):
        cluster_pre[i]=[]
        cluster_aug[i]=[]
        cluster_last[i]=[]
        cluster_y[i]=[]
        cluster_yt[i]=[]

    clusters = np.array(kmeans.labels_)
    len_last = len(x_last)

    for i in range(len_last):
        cluster_last[clusters[i]].append(x_last[i])
        cluster_y[clusters[i]].append(y_all[i])
        cluster_yt[clusters[i]].append(yt_all[i])
    for i in range(len_last, len(x_all)):
        cluster_pre[clusters[i]].append(x_pre[i-len_last])
        cluster_aug[clusters[i]].append(aug_val[i-len_last])
    for i in range(n_cluster):
        cluster_pre[i]=np.array(cluster_pre[i])
        cluster_aug[i]=np.array(cluster_aug[i])
        cluster_last[i]=np.array(cluster_last[i])
        cluster_y[i]=np.array(cluster_y[i])
        cluster_yt[i]=np.array(cluster_yt[i])
    return cluster_pre, cluster_aug, cluster_last, cluster_y, cluster_yt, clusters
