from typing import Dict

import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from sklearn import model_selection as sk_ms
from sklearn import multiclass as sk_mc
from sklearn import preprocessing as sk_prep
from collections import Counter

import time
import torch.nn as nn
from torch.optim import Adam


import torch
from torch_geometric.data import Data
from sklearnex import patch_sklearn 
patch_sklearn('KMeans')



def cluster_eval(y_true, y_pred):
    """code source: https://github.com/bdy9527/SDCN"""
    y_true = y_true.detach().cpu().numpy() if type(y_true) is torch.Tensor else y_true
    y_pred = y_pred.detach().cpu().numpy() if type(y_pred) is torch.Tensor else y_pred

    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    # print(f"INFO {l1}, {l2}")
    # print(f"INFO numclasses {numclass1}, {numclass2}")
    # fill out missing classes
    ind = 0
    c2 = Counter(y_pred)
    maxclass = sorted(c2.items(), key=lambda item: item[1], reverse=True)[0][0]
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                ind = y_pred.tolist().index(maxclass)
                y_pred[ind] = i

    l2 = list(set(y_pred))
    numclass2 = len(l2)
    # print(f"INFO filled numclasses {numclass1}, {numclass2}")

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        # print(f"INOF: {len(l2)}, {len(indexes)}, {i}")
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def unsup_eval(y_true, y_pred, epoch=0, quiet=False):
    y_true = y_true.detach().cpu().numpy() if type(y_true) is torch.Tensor else y_true
    y_pred = y_pred.detach().cpu().numpy() if type(y_pred) is torch.Tensor else y_pred

    acc, f1 = cluster_eval(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    if not quiet:
        print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def kmeans_test(X, y, n_clusters, repeat=10, epoch=0, quiet=True):
    y = y.detach().cpu().numpy() if type(y) is torch.Tensor else y
    X = X.detach().cpu().numpy() if type(X) is torch.Tensor else X

    mask_nan = np.isnan(X)
    mask_inf = np.isinf(X)
    X[mask_nan] = 1
    X[mask_inf] = 1

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    for _ in range(repeat):


        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)



        acc_score, nmi_score, ari_score, macro_f1 = unsup_eval(
            y_true=y, y_pred=y_pred,
            epoch=epoch, quiet=quiet
        )
        acc_list.append(acc_score)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
        f1_list.append(macro_f1)
    return np.mean(acc_list), np.std(acc_list), np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(
        ari_list), np.mean(f1_list), np.std(f1_list)


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


def LRE(x, y, idx_train, idx_val, idx_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = x.detach().to(device)
    input_dim = x.size()[1]
    y = y.detach().to(device)
    num_classes = y.max().item() + 1
    classifier = LogisticRegression(input_dim, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    output_fn = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()

    best_val_micro = 0
    best_test_micro = 0
    best_test_macro = 0
    best_epoch = 0
    num_epochs = 500
    test_interval = 20


    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(x[idx_train])
        loss = criterion(output_fn(output), y[idx_train])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % test_interval == 0:
            classifier.eval()
            y_test = y[idx_test].detach().cpu().numpy()
            y_pred_test = classifier(x[idx_test]).argmax(-1).detach().cpu().numpy()
            test_micro = f1_score(y_test, y_pred_test, average='micro')
            test_macro = f1_score(y_test, y_pred_test, average='macro')

            y_val = y[idx_val].detach().cpu().numpy()
            y_pred_val = classifier(x[idx_val]).argmax(-1).detach().cpu().numpy()
            val_micro = f1_score(y_val, y_pred_val, average='micro')

            if val_micro > best_val_micro:
                best_val_micro = val_micro
                best_test_micro = test_micro
                best_test_macro = test_macro
                best_epoch = epoch


    print('best_test_epoch:', best_epoch)
    return {
        'micro_f1': best_test_micro,
        'macro_f1': best_test_macro
        }
