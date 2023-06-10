import torch
import torch.nn.functional as F
import copy



def edge_drop(edge_index, p=0.4):
    # copy edge_index
    edge_index = copy.deepcopy(edge_index)
    num_edges = edge_index.size(1)
    num_droped = int(num_edges*p)
    perm = torch.randperm(num_edges)

    edge_index = edge_index[:, perm[:num_edges-num_droped]]
    
    return edge_index



def feat_drop(x, p=0.2):
    # copy x
    x = copy.deepcopy(x)
    mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x[:, mask] = 0

    return x

