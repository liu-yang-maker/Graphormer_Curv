# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
sys.path.append("/mnt/sfs_turbo/liuyang/Lai/Graphormer/graphormer/data")

import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache
import pyximport
import torch.distributed as dist

import scipy.sparse as sp

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

import networkx as nx
from LocalRicci import compute_ricci_locally as Ricci

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()


    ################################################## Curv

    # G = nx.from_edgelist(item['edge_index'].numpy().T)
    # curv = load_curv(G, True, None)

    # data_list = []
    # for i in range(len(curv)):
    #     data_list.append(curv[i][1])

    # row = []
    # col = []
    # for i in range(len(curv)):
    #     row.append(curv[i][0][0])
    #     col.append(curv[i][0][1])
    # sym_row = row + col
    # sym_col = col + row

    # sym_list = data_list + data_list

    # sampled_adj = sp.coo_matrix((sym_list, (sym_row, sym_col)), shape=spatial_pos.shape).todense()
    # spatial_pos = torch.from_numpy(sampled_adj.A)
    # spatial_pos = spatial_pos.long()
    ##################################################

    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        dist.barrier()

    def process(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        dist.barrier()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        return preprocess_item(item)



def load_curv(G:nx.Graph, compute_curv:bool, curv_file:str): 
        if compute_curv:
            # print('Computing Ricci curvature...')
            G.remove_edges_from(nx.selfloop_edges(G))
            # print(G.number_of_edges())
            rc_G = Ricci(G)
            rc_G.compute_ricci_curvature_edges(G.edges())
            edge_curvature = list(nx.get_edge_attributes(rc_G.G, 'ricciCurvature').items())
            curv = sorted(edge_curvature, key=lambda x:x[1])
            curv_list = []
            for (x, y), cur in curv:
                curv_list.append('{} {} {:.6f}'.format(x, y, cur))
            
            if curv_file is not None:
                with open(curv_file, 'w') as f:
                    f.write('\n'.join(curv_list))
        else:
            lines = open(curv_file, 'r').readlines()
            curv = []
            for line in lines:
                x, y, cur = line.strip().split()
                curv.append(((int(x),int(y)), float(cur)))
        # print('Ricci curvature loadded.')
        return curv