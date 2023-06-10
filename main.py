import argparse
import math
import random
import os
import time

import numpy as np

from transforms import *

from dataset import load_data
from eval_tools import kmeans_test, LRE

import torch


import scipy.sparse as sp
import torch.nn as nn
from torch.nn.parameter import Parameter
import networkx as nx
from scipy.sparse import csr_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix


import csv
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score

import torch_geometric.transforms as T





def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_mlp', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--wd_mlp', type=float, default=0.0005)
parser.add_argument('--drop', type=float, default=0.2)
parser.add_argument('--dropout_mlp', type=float, default=0.4)
parser.add_argument('--alpha1', type=float, default=0.01)
parser.add_argument('--alpha2', type=float, default=0.01)
parser.add_argument('--alpha3', type=float, default=100)
parser.add_argument('--alpha4', type=float, default=1)
parser.add_argument('--gate', type=int, default=30)
parser.add_argument('--homo', type=float, default=0.3)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=0)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--output_size', type=int, default=512)
parser.add_argument('--feat_drop', type=float, default=0.2)
parser.add_argument('--edge_drop', type=float, default=0.2)
parser.add_argument('--tau', type=float, default=0.5)

parser.add_argument('--dataset', type=str, default='cornell')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--task', type=str, default='node_classification',
                    help='node_cluster/node_classification')
parser.add_argument('--filename', type=str, default='default')



args = parser.parse_args()

seed_it(args.seed)

print(args)
torch.cuda.set_device(args.gpu)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

features, edges, num_classes, train_mask, val_mask, test_mask, labels, num_nodes, feat_dimension = load_data(args.dataset)



if len(labels.shape) == 1:
    labels = labels.unsqueeze(1)

edges, features = edges.to(device), features.to(device)

print(f"num nodes {num_nodes} | num classes {num_classes} | num node feats {feat_dimension}")

class MLP(nn.Module):
    def __init__(self, in_channels, dropout_mlp, gate_channels):
        super(MLP, self).__init__()
        self.dropout = dropout_mlp

        self.gate1 = nn.Linear(in_channels, gate_channels).requires_grad_(requires_grad=True) 
        self.gate2 = nn.Linear(in_channels, gate_channels).requires_grad_(requires_grad=True) 
        self.gate3 = nn.Linear(2*gate_channels+1, 1).requires_grad_(requires_grad=True)
        self.sigm = nn.Sigmoid()



        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.gate1.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate2.weight, gain=1.414)
        nn.init.xavier_normal_(self.gate3.weight, gain=1.414)


    def forward(self, h1, h2, deg):

        z1 = self.gate1(h1).squeeze()
        z1 = F.dropout(z1, p=self.dropout, training=self.training)


        z2 = self.gate2(h2).squeeze()
        z2 = F.dropout(z2, p=self.dropout, training=self.training)
        

        w1 = torch.cat((z1, z2, deg), dim=1)

        output = self.gate3(w1)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.sigm(output)
        return output


class Model(nn.Module):
    def __init__(self, input, hidden, output, tau, drop, dropout_mlp, gate, batch_size):
        super(Model, self).__init__()

        self.tau = tau
        self.drop = drop

        self.fcs  = nn.ModuleList()
        self.fcs.append(GCN(input, 2 * hidden))
        self.fcs.append(GCN(2 * hidden, hidden))
        self.fcs.append(torch.nn.Linear(hidden, output))
        self.fcs.append(torch.nn.BatchNorm1d(output))
        self.fcs.append(torch.nn.Linear(output, hidden))


        self.params = list(self.fcs.parameters())

        self.mlp = nn.ModuleList()
        self.mlp.append(MLP(hidden, dropout_mlp, gate))
        self.params_mlp = list(self.mlp.parameters())
        self.activation = F.relu
        self.batch_size = batch_size


    def forward(self, x1, x2, adj1, adj2, drop_edge_index, deg):
        x1 = F.dropout(x1, p=self.drop, training=self.training)
        x2 = F.dropout(x2, p=self.drop, training=self.training)

        h1 = self.activation(self.fcs[0](x1, adj1))
        h1 = self.fcs[1](h1, adj1)


        h2 = self.activation(self.fcs[0](x2, adj1))
        h2 = self.fcs[1](h2, adj1)

        h3 = self.activation(self.fcs[0](x1, adj2))
        h3 = self.fcs[1](h3, adj2)

        h4 = self.activation(self.fcs[0](x1, drop_edge_index))
        h4 = self.fcs[1](h4, drop_edge_index)

        beta = self.mlp[0](h1.detach(), h3.detach(), deg.detach())
        return h1, h2, h3, h4, beta




    def projection(self, z):
        z = F.elu(self.fcs[3](self.fcs[2](z)))
        return self.fcs[4](z)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))





    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)


        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


    def cos_loss(self, X, N, beta):
        S = F.cosine_similarity(X, N).unsqueeze(1)
        simi = torch.matmul(beta.t(), S)
        return simi, S



    @torch.no_grad()
    def get_embedding(self, x1, adj1, adj2, deg):
        h1 = self.activation(self.fcs[0](x1, adj1))
        h1 = self.fcs[1](h1, adj1)

        h3 = self.activation(self.fcs[0](x1, adj2))
        h3 = self.fcs[1](h3, adj2)

        beta = self.mlp[0](h1.detach(), h3.detach(), deg.detach())
        return h1 + torch.mul(beta.detach(), h3)





def train(model, optimizer, x1, x2, adj1, adj2, drop_edge_index, args, deg):
    model.train()

    optimizer.zero_grad()

    h1, h2, h3, h4, beta = model(x1, x2, adj1, adj2, drop_edge_index, deg)


    ### train gnn
    loss1 = model.loss(h1, h2)
    loss2 = model.loss(h3, h4)

    combined1 = h1 + torch.mul(beta.detach(), h3)
    combined2 = h2 + torch.mul(beta.detach(), h4)
    loss3 = model.loss(combined1, combined2)
    loss_gnn = loss1 + args.alpha1 * loss2 + args.alpha2 * loss3

    print(f"loss_gnn: {loss_gnn.item()}")
    

    loss_gnn.backward()
    optimizer.step()
 
    optimizer.zero_grad()


    ### train mlp
    h1, h2, h3, h4, beta = model(x1, x2, adj1, adj2, drop_edge_index, deg)


    hetero_loss, S = model.cos_loss(h1.detach(), h3.detach(), beta)
    L2_loss = torch.norm(beta, dim = 0)
    reg = torch.abs(torch.mean(beta) - args.homo)


    mlp_loss = hetero_loss + args.alpha3 * reg + args.alpha4 * L2_loss

    print(f"hetero_loss: {hetero_loss.item()}")
    print(f"reg: {reg.item()}")
    print(f"mlp_loss: {mlp_loss.item()}")
    print(f"l2_loss: {L2_loss.item()}")
    mlp_loss.backward()

    optimizer.step()
    return loss_gnn.item(), mlp_loss.item(), beta, S



def test(model, labels, epoch, x1, adj1, adj2, deg, idx_train, idx_val, idx_test, task='node_classification'):

    model.eval()

    with torch.no_grad():
        representations = model.get_embedding(x1, adj1, adj2, deg)
        labels = labels.squeeze(1)

    


    if task == 'node_classification':



        result = LRE(representations, labels, idx_train, idx_val, idx_test)



        print(f"micro_f1: {result['micro_f1']}")
        print(f"macro_f1: {result['macro_f1']}")
    elif task == 'node_cluster':
        result = result = kmeans_test(representations, labels, n_clusters=num_classes, repeat=1)
        
        print(f'Epoch: {epoch:02d}, '
                    f'acc: {100 * result[0]:.2f}%, '
                    f'nmi: {100 * result[2]:.2f}%, '
                    f'ari: {100 * result[4]:.2f}%, ')

    return result






input = feat_dimension
hidden = args.hidden_size
output = args.output_size
tau = args.tau
drop = args.drop
dropout_mlp = args.dropout_mlp
gate = args.gate
batch_size = args.batch_size




best_results = []
for run in range(args.runs):

    cur_split = 0 if (train_mask.shape[1]==1) else (run % train_mask.shape[1])

    idx_train = train_mask[:, cur_split]
    dx_val = val_mask[:, cur_split]
    idx_test = test_mask[:, cur_split]
    idx_train = np.where(idx_train == 1)[0]
    idx_val = np.where(dx_val == 1)[0]
    idx_test = np.where(idx_test == 1)[0]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)




    model = Model(input, hidden, output, tau, drop, dropout_mlp, gate, batch_size)
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam([
                            {'params':model.params, 'lr':args.lr, 'weight_decay':args.weight_decay},
                            {'params':model.params_mlp,'weight_decay':args.wd_mlp,'lr':args.lr_mlp},
                            ])

    all_results = []

    best_micro = 0
    best_macro = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        x1, x2 = features, feat_drop(features, p=args.feat_drop)
        edge_index = edges


        adj1 = torch.eye(num_nodes).to(device)
        
        adj = to_scipy_sparse_matrix(edge_index=edges, num_nodes = num_nodes)
        adj2 = sparse_mx_to_torch_sparse_tensor(adj).to(device)

        deg = sp.coo_matrix.sum(adj, axis=1)
        deg = torch.tensor(deg)
        deg = deg.to(device)


        drop_edge_index = edge_drop(edge_index, p=args.edge_drop)
        drop_edge_index = to_scipy_sparse_matrix(edge_index=drop_edge_index, num_nodes = num_nodes)
        drop_edge_index = sparse_mx_to_torch_sparse_tensor(drop_edge_index).to(device)


        gnnloss, mlploss, beta, similarity = train(model, optimizer, x1, x2, adj1, adj2, drop_edge_index, args, deg)
        print(f"run: {run}, epoch: {epoch}")
        result = test(model, labels, epoch, x1, adj1, adj2, deg, idx_train, idx_val, idx_test, task=args.task)



        if args.task == 'node_classification':
            if result['micro_f1'] > best_micro:
                best_micro = result['micro_f1']
                best_macro = result['macro_f1']
                best_epoch = epoch

        elif args.task == 'node_cluster':
            all_results.append(result)

    
    if args.task == 'node_classification':
        print('run:', run)
        print('best_epoch:', best_epoch)
        print('best_micro:', best_micro)
        best_results.append((best_micro, best_macro))

    elif args.task == 'node_cluster':
        result = torch.tensor(all_results)

        best_acc = result[:, 0].max()
        best_nmi = result[:, 2].max()
        best_ari = result[:, 4].max()

        print(f'Highest acc: {100*result[:, 0].max():.2f}')
        print(f'Highest nmi: {100*result[:, 2].max():.2f}')
        print(f'Highest ari: {100*result[:, 4].max():.2f}')

        best_results.append((best_acc, best_nmi, best_ari))
    else:
        raise ValueError("Unrecognized task")





if args.task == 'node_classification':
    best_result = 100*torch.tensor(best_results)
    best_test_micro_f1 = best_result[:, 0]
    best_test_macro_f1 = best_result[:, 1]

    print(f"test micro-f1: {best_test_micro_f1.mean():.2f} ± {best_test_micro_f1.std():.2f}," +
          f"test macro-f1: {best_test_macro_f1.mean():.2f} ± {best_test_macro_f1.std():.2f}, " +
          f"{args.__repr__()}\n")




elif args.task == 'node_cluster':
    best_result = 100*torch.tensor(best_results)
    best_acc = best_result[:, 0]
    best_nmi = best_result[:, 1]
    best_ari = best_result[:, 2]
    print(f"acc: {best_acc.mean():.3f} ± {best_acc.std():.3f}," +
          f"nmi: {best_nmi.mean():.3f} ± {best_nmi.std():.3f}," +
          f"ari: {best_ari.mean():.3f} ± {best_ari.std():.3f}, " +
          f"{args.__repr__()}\n")

