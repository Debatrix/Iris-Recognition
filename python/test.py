# coding=utf-8
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


def select_to_show(features, labels, num=10):
    from random import choices
    select = []
    while len(select) < num:
        select_item = choices(labels)[0]
        select_pair = select_item[:-1] + 'R' if select_item[
            -1] == 'L' else select_item[:-1] + 'L'
        select += [
            select_item, select_pair
        ] if select_pair in labels and select_item not in select else []
    select_index = [idx for idx, label in enumerate(labels) if label in select]
    select_features = [features[idx] for idx in select_index]
    select_labels = [labels[idx] for idx in select_index]
    return select_features, select_labels


def decomposition(labels, features, show=True):
    labels_set = list(set(labels))

    features = PCA(n_components=2).fit_transform(features)
    # features = TSNE(n_components=2).fit_transform(features)

    for lid, label in enumerate(labels_set):
        c = np.random.random(3)
        idxs = [idx for idx, sl in enumerate(labels) if sl == label]
        feature = np.zeros((len(idxs), 2))
        for x, y in enumerate(idxs):
            feature[x, :] = features[y, :]
        plt.scatter(feature[:, 0], feature[:, 1], color=c, label=label)
    plt.legend(ncol=2)
    plt.title("Clustering Result")
    plt.show()


def plot_DET(features, labels, dist=None):
    num = features.shape[0]
    sim = np.dot(features, features.T) if dist is None else dist
    sim = np.delete(np.reshape(sim, (1, -1)), [x * x for x in range(num)])
    bool = np.dot(labels, labels.T)
    bool = np.delete(np.reshape(bool, (1, -1)), [x * x for x in range(num)])

    fpr, tpr, thresholds = metrics.roc_curve(bool, sim)
    fnr = 1 - tpr
    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        fnr,
        color='darkorange',
        lw=lw,
        label='auc = {:.2f}, eer = {:.2e}'.format(roc_auc, eer))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('DET curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    show_flag = True
    dataset = 'CASIA-Iris-Thousand'
    with open(os.path.join('./feature/', dataset + '.pkl'), 'rb') as f:
        ft_load = pickle.load(f)
        onehot = ft_load['onehot']
        features = ft_load['features']
        labels = ft_load['labels']
        hm_dists = ft_load['hm_dists']

    silhouette_score = metrics.silhouette_score(
        features, labels, metric='euclidean')
    calinski_harabaz_score = metrics.calinski_harabaz_score(features, labels)
    print(
        "Clustering Result\n\tsilhouette score:{}\n\tcalinski harabaz score:{}\n\n"
        .format(silhouette_score, calinski_harabaz_score))
    if show_flag:
        features, labels = select_to_show(features, labels)
        decomposition(labels, features)
    else:
        plot_DET(features, onehot, dist=hm_dists)