# -*- coding: utf-8 -*-
from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
from collections.abc import Sequence
import math
PI = math.pi
import torch
import random
from torch import optim
from torch.utils.data import DataLoader
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
from rde.modules.encoders.single import PerResidueEncoder,AAEmbedding
from rde.utils.transforms.noise import set_chis
from rde.utils.transforms.noise import ChiSelection,remove_by_chi
from copy import deepcopy
from collections import Counter
from .rde import CircularSplineRotamerDensityEstimator
from scipy import stats, special

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.relpos_embed = nn.Embedding(2 * max_relative_feature + 1, num_embeddings)
        self.mask_embed = nn.Embedding(2, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        E = self.relpos_embed(d) + self.mask_embed(mask)
        return E

class ContactEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(ContactEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def gaussian(x, mean, std):
    pi = 3.1415926
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

class GaussianLayer(nn.Module):
    def __init__(self, K=16, edge_types=5*5, max_aa_types=22):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)  # 维度 = K
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)        # 维度 = 1
        self.bias = nn.Embedding(edge_types, 1)       # padding_idx: 标记输入中的填充值

        self.max_aa_types = max_aa_types
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, self.K, padding_idx=441)

    def gather_nodes(self, nodes, neighbor_idx):
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
        return neighbor_features
    def forward(self, aa, X, E_idx, mask_atoms, mask_attend):
        # Pair identities[氨基酸类型pair编码]
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)

        aa_pair_neighbor = torch.gather(aa_pair, 2, E_idx)
        feat_aapair = self.aa_pair_embed(aa_pair_neighbor.to(torch.long))

        n_complex, n_residue, n_atom, _ = X.shape

        X_views = X.view((X.shape[0], X.shape[1], -1))
        neighbors_X = self.gather_nodes(X_views, E_idx)
        neighbors_X = neighbors_X.view(X.shape[0], X.shape[1], -1, 5, 3)
        delta_pos = neighbors_X.unsqueeze(-2) - X.unsqueeze(2).unsqueeze(-3)  #　修改为邻居节点到Ｘ所有原子的距离

        D_A_B_neighbors = delta_pos.norm(dim=-1).view(n_complex, n_residue, E_idx.shape[-1], n_atom, n_atom).to(X.device)
        edge_types = torch.arange(0, n_atom*n_atom).view(n_atom, n_atom).to(X.device)
        mul = self.mul(edge_types).squeeze(-1)   # 边类型嵌入，然后对边求和， gamma
        bias = self.bias(edge_types).squeeze(-1)    # 边类型嵌入，然后对边求和, beta

        x = mul * D_A_B_neighbors+ bias
        mask_atoms_attend = self.gather_nodes(mask_atoms, E_idx)
        mask_attend_new = (mask_atoms_attend[:, :, :, None, :] * mask_atoms_attend[:, :, :, :, None])
        x = x * mask_attend_new
        x = x.unsqueeze(-1).expand(-1, -1, -1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        gbf = gaussian(x.float(), mean, std).type_as(self.means.weight).view(n_complex, n_residue, -1,n_atom*n_atom*self.K)
        return gbf * mask_attend.unsqueeze(-1), feat_aapair * mask_attend.unsqueeze(-1)

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, patch_size, seq_neighbours, seq_nonneighbours=3, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.patch_size = patch_size
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.seq_neighbours = seq_neighbours
        self.seq_nonneighbours = seq_nonneighbours

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25 + 7 * 2 +16
        self.gbf = GaussianLayer(K=16, edge_types=5*5)
        self.gbf_proj = NonLinear(input=edge_in, output_size=edge_features)

        self.dropout = nn.Dropout(0.1)

    def _dist(self, X, mask, residue_idx, eps=1E-6):
        """ Pairwise euclidean distances """
        N = X.size(1)
        # 序列最近邻
        residue_offset = residue_idx[:, :, None] - residue_idx[:, None, :]  # 氨基酸顺序偏移量
        sequence_idx = torch.arange(self.patch_size, dtype=torch.long).unsqueeze(0).expand(X.size(0),-1).to(X.device)
        sequence_offset = sequence_idx[:, :, None] - sequence_idx[:, None, :]

        offset = ((torch.abs(residue_offset) < 50) & (torch.abs(sequence_offset) <= self.seq_neighbours)).float()  # mask self
        D_sequence  = residue_idx[:, None, :] * offset  # 使得mask残基对齐
        offset_neighbors, E_idx_sequential = torch.topk(D_sequence,
                                                       np.minimum(self.seq_neighbours * 2 + 1, offset.shape[-1]),
                                                       dim=-1, largest=True) # 取最大值
        mask_sequential = offset_neighbors > 0.0

        # 空间最近邻
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (~mask_2D) * D_max

        _, E_idx_spatial = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)  # 取最小值
        mask_spatial = gather_edges(mask_2D.unsqueeze(-1), E_idx_spatial)[:,:,:,0]

        # 序列远空间近
        mask_2D = (offset == 0.)  # 序列远
        D =  mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (~mask_2D) * D_max
        _, E_idx_nonsequential = torch.topk(D_adjust, np.minimum(self.seq_nonneighbours, X.shape[1]), dim=-1, largest=False)
        mask_nonsequential = gather_edges(mask_2D.unsqueeze(-1), E_idx_nonsequential)[:, :, :, 0]

        return E_idx_spatial, mask_spatial, E_idx_sequential, mask_sequential,E_idx_nonsequential,mask_nonsequential

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q
    def _orientations_coarse(self, X, E_idx, mask_attend):
        # Pair features
        u = torch.ones_like(X)
        u[:,1:,:] = X[:, 1:, :] - X[:,:-1,:]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(X)
        b[:, :-1,:] = u[:, :-1,:] - u[:, 1:,:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(X)
        n[:,:-1,:] = torch.cross(u[:,:-1,:], u[:,1:,:])
        n = F.normalize(n, dim=-1)
        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=2)
        local_frame = local_frame.view(list(local_frame.shape[:2]) + [9])

        X_neighbors = gather_nodes(X, E_idx)
        O_neighbors = gather_nodes(local_frame, E_idx)
        # Re-view as rotation matrices
        local_frame = local_frame.view(list(local_frame.shape[:2]) + [3, 3])    # Oi
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])    # Oj
        # # # Rotate into local reference frames ，计算最近邻相对x_i的局部坐标系
        t = X_neighbors - X.unsqueeze(-2)
        t = torch.matmul(local_frame.unsqueeze(2), t.unsqueeze(-1)).squeeze(-1)  # 边特征第二项
        t = F.normalize(t, dim=-1) * mask_attend.unsqueeze(-1)
        r = torch.matmul(local_frame.unsqueeze(2).transpose(-1, -2), O_neighbors)  # 边特征第三项
        r = self._quaternions(r)  * mask_attend.unsqueeze(-1)   # 边特征第三项

        return torch.cat([t, r, 1 - 2 * t.abs(), 1 - 2 * r.abs()], dim=-1)

    def PerEdgeEncoder(self, X, E_idx, mask_attend, residue_idx, chain_labels):
        Ca = X[:, :, 1]
        # 1. Relative spatial encodings
        O_features = self._orientations_coarse(Ca, E_idx, mask_attend)

        # 2. 位置编码
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]
        # d_chains：链内和链间标记，为1表示链内
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)

        return E_positional, O_features

    def _set_Cb_positions(self, X, mask_atom):
        """
        Args:
            pos_atoms:  (L, A, 3)
            mask_atoms: (L, A)
        """
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1] - X[:, :, 0]
        c = X[:, :, 2] - X[:, :, 1]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1]  # 虚拟Cb原子
        X[:, :, 4] = mask_atom[:, :, 4].unsqueeze(-1) * X[:, :, 4] + \
                     (~mask_atom[:, :, 4].unsqueeze(-1)) * Cb  # Cb缺失的按虚拟计算
        return X * mask_atom[:,:,:,None]

    def forward(self, batch):
        X = batch["pos_atoms"]
        mask_atom = batch["mask_atoms"]   # N = 0; CA = 1; C = 2; O = 3; CB = 4;
        residue_idx = batch["residue_idx"]
        chain_labels = batch["chain_nb"]     # # d_chains：链内和链间标记，为1表示链内
        mask_residue = batch["mask"]
        aa = batch["aa"]

        X = self._set_Cb_positions(X, mask_atom)
        Cb = X[:, :, 4, :]

        # 这里考虑用Cb原子来计算最短距离
        E_idx_spatial, mask_spatial, E_idx_sequential, mask_sequential,E_idx_nonsequential,mask_nonsequential = self._dist(Cb, mask_residue, residue_idx)

        # spactial coding & sequential coding
        E_idx = torch.cat([E_idx_spatial,E_idx_nonsequential, E_idx_sequential], dim=-1)
        mask_attend = torch.cat([mask_spatial,mask_nonsequential, mask_sequential], dim=-1)
        mask_attend = mask_residue.unsqueeze(-1) * mask_attend
        E_positional, O_features = self.PerEdgeEncoder(X, E_idx, mask_attend, residue_idx, chain_labels)

        # 2. 高斯核编码
        gbf_feature, aapair_feature = self.gbf(aa, X, E_idx, mask_atom, mask_attend)

        E = torch.cat((E_positional, gbf_feature, O_features, aapair_feature), dim=-1)
        E = self.gbf_proj(E) * mask_attend.unsqueeze(-1)

        idx_spatial = self.top_k + self.seq_nonneighbours
        idx_seq_neighbours = 2 * self.seq_neighbours + 1
        # idx_spatial = self.top_k  # nonneighbours[no]

        E_spatial = E[...,:idx_spatial,:]
        E_idx_spatial = E_idx[...,:idx_spatial]
        mask_spatial = mask_attend[...,:idx_spatial]
        E_sequential = E[..., -idx_seq_neighbours:, :]
        E_idx_sequential = E_idx[..., -idx_seq_neighbours:]
        mask_sequential = mask_attend[..., -idx_seq_neighbours:]

        return E_spatial, E_idx_spatial, mask_spatial, E_sequential, E_idx_sequential, mask_sequential

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, encoder_normalize_before=True, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden) for i in range(3)])

        self.normalize_before = encoder_normalize_before

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        # ReZero is All You Need: Fast Convergence at Large Depth
        self.resweight = nn.Parameter(torch.Tensor([0]))

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(0, h_V, before=True)

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)         # h_j || edge
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)   # h_i
        h_EV = torch.cat([h_V_expand, h_EV], -1)  # h_i || h_j || edge
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        # ReZero
        dh = dh * self.resweight
        h_V = residual + self.dropout1(dh)
        h_V = self.maybe_layer_norm(0, h_V, after=True)

        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(1, h_V, before=True)

        # Position-wise feedforward
        dh = self.dense(h_V)
        # ReZero
        dh = dh * self.resweight
        h_V = residual + self.dropout2(dh)
        h_V = self.maybe_layer_norm(1, h_V, after=True)

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # assert not torch.any(torch.isnan(h_V)),'h_V nan exists!'
        residual = h_E
        h_E = self.maybe_layer_norm(2, h_E, before=True)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)  # h_j || edge
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)  # h_i
        h_EV = torch.cat([h_V_expand, h_EV], -1)   # h_i || h_j || edge
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = residual + self.dropout3(h_message)
        h_E = self.maybe_layer_norm(2, h_E, after=True)

        return h_V, h_E

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, decoder_normalize_before=True, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(num_hidden) for i in range(2)])

        self.normalize_before = decoder_normalize_before

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(0, h_V, before=True)
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        # ReZero
        dh = self.dropout1(dh) * self.resweight
        h_V = residual + dh
        h_V = self.maybe_layer_norm(0, h_V, after=True)

        # pre-norm
        residual = h_V
        h_V = self.maybe_layer_norm(1, h_V, before=True)
        # Position-wise feedforward
        dh = self.dense(h_V)
        # ReZero
        dh = self.dropout1(dh) * self.resweight
        h_V = residual + dh
        h_V = self.maybe_layer_norm(1, h_V, after=True)

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

class OutLayer(nn.Module):
    def __init__(self, num_in, num_hidden,  dropout=0.1, num_heads=None, scale=30):
        super(OutLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_S, h_V, h_E, E_idx, mask_attend, mask_V=None):
        """ Parallel computation of full transformer layer """
        # Concatenate h_V_i to h_E_ij
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)  # h_j || h_edge
        h_EXV_out = cat_neighbors_nodes(h_V, h_ES, E_idx)  # h_j(enc) || h_j || h_edge
        h_EXV_out = mask_attend.unsqueeze(-1) * h_EXV_out

        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1, h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EXV_out], -1)   # h_i(enc) ||  h_j(enc) || h_j || h_edge
        h_message = self.W2(self.act(self.W1(h_EV)))
        dh = torch.sum(h_message, -2) / self.scale
        h_V = h_V + self.dropout1(dh)
        h_V = self.norm1(h_V)
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

def init_params(module):
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        # data.copy_(
        #     data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        # )
        if data.dim() > 1:
            nn.init.xavier_uniform_(data)

    if isinstance(module, GaussianLayer):
        nn.init.uniform_(module.means.weight, 0, 3)
        nn.init.uniform_(module.stds.weight, 0, 3)
        nn.init.constant_(module.bias.weight, 0)
        nn.init.constant_(module.mul.weight, 1)
        normal_(module.aa_pair_embed.weight.data)
        if module.aa_pair_embed.padding_idx is not None:
            module.aa_pair_embed.weight.data[module.aa_pair_embed.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        # 初始化层归一化的权重为 1，偏置为 0
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

class ProteinMPNN_NET(nn.Module):
    NUM_CHI_ANGLES = 4
    def __init__(self, cfg, num_letters=21, readout="sum", train_chi_id=None):
        super(ProteinMPNN_NET, self).__init__()

        # Hyperparameters
        self.node_features = cfg.node_features
        self.edge_features = cfg.edge_features
        self._replace_rate = cfg._replace_rate
        self.num_encoder_layers = cfg.num_encoder_layers
        self.dropout = cfg.dropout
        self.patch_size = cfg.patch_size
        self._mask_token_rate = 1 - self._replace_rate
        hidden_dim = cfg.hidden_dim
        self.num_rbf = 16
        self.seq_neighbours = cfg.seq_neighbours
        self.seq_nonneighbours = cfg.seq_nonneighbours

        self.features = ProteinFeatures(cfg.edge_features, cfg.node_features, cfg.patch_size,  top_k=cfg.k_neighbors, augment_eps=cfg.augment_eps, seq_neighbours = self.seq_neighbours,seq_nonneighbours= cfg.seq_nonneighbours)
        self.W_e = nn.Linear(cfg.edge_features, hidden_dim, bias=True)
        self.W_es = nn.Linear(cfg.edge_features, hidden_dim, bias=True)

        # # Residue Encoding
        self.single_encoders = nn.ModuleList([
            PerResidueEncoder(
                feat_dim=cfg.encoder.node_feat_dim,
                max_num_atoms=5,  # N, CA, C, O, CB,
            )
            for _ in range(3)
        ])
        self.embeddinges = nn.ModuleList([
            AAEmbedding(feat_dim=cfg.encoder.node_feat_dim, infeat_dim=123)
            for _ in range(3)
        ])

        self.masked_biases = nn.ModuleList([
            nn.Embedding(
                num_embeddings=2,
                embedding_dim=cfg.encoder.node_feat_dim,
                padding_idx=0,
            )
            for _ in range(3)
        ])

        # Encoder layers
        self.encoder_layers_spatial = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])
        self.encoder_layers_sequential = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])

        self.single_fusion = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim)
        )
        # self.single_fusion = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*2, dropout=cfg.dropout)
            for _ in range(cfg.num_decoder_layers)
        ])

        self.enc_mask_token = nn.Parameter(torch.zeros(1, cfg.encoder.node_feat_dim))  # 可学习参数

        self.out_layer = OutLayer(hidden_dim*3, hidden_dim, dropout=cfg.dropout)
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        self.enc_centrality = nn.Parameter(torch.zeros(1, 2*cfg.seq_neighbours+1+cfg.k_neighbors+cfg.seq_nonneighbours))
        # self.enc_centrality = nn.Parameter(torch.zeros(1, 2 * cfg.seq_neighbours + 1 + cfg.k_neighbors))
        # self.enc_centrality = nn.Parameter(torch.zeros(1, cfg.k_neighbors + cfg.seq_nonneighbours))

        # Pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.smoothL1Loss = nn.SmoothL1Loss()
        self.logcoshLoss = LogCoshLoss()

        self.apply(lambda module: init_params(module))

    def gather_centrality(self, nodes, neighbor_idx, mask):
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:2] + [-1])
        # 　增加可学习参数
        neighbor_features = neighbor_features+ self.enc_centrality.unsqueeze(0)
        neighbor_centrality = torch.sum(neighbor_features, dim=-1) * mask

        neighbor_centrality = (neighbor_centrality / torch.max(neighbor_centrality, dim=-1)[0].unsqueeze(-1) )* mask

        return neighbor_centrality

    def dihedral_encode(self, batch, chi_flag, code_idx):
        mask_residue = batch['mask']
        if chi_flag == True:
            chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]
        else:
            chi = batch['chi']

        x = self.single_encoders[code_idx](
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,
        )
        # 氨基酸极性编码
        seq_emb = self.embeddinges[code_idx](batch['aa'])

        b = self.masked_biases[code_idx](batch['mut_flag'].long())  # # 仅包括ratio-mutation残基
        x = seq_emb + x + b # (6,128,128)

        return x

    def encode(self, batch):
        # 编码器
        E_spatial, E_idx_spatial, mask_spatial, E_sequential, E_idx_sequential, mask_sequential = self.features(batch)
        h_E_spatial = self.W_e(E_spatial)
        h_E_sequential = self.W_es(E_sequential)

        h_V_spatial = self.dihedral_encode(batch, chi_flag=True, code_idx=0)
        # Encoder
        mask =  batch['mask']
        for i, layer in enumerate(self.encoder_layers_spatial):
            h_V_spatial, h_E_spatial = layer(h_V_spatial, h_E_spatial, E_idx_spatial, mask, mask_spatial)

        h_V_sequential = self.dihedral_encode(batch, chi_flag=True, code_idx=1)
        for i, layer in enumerate(self.encoder_layers_sequential):
            h_V_sequential, h_E_sequential = layer(h_V_sequential, h_E_sequential, E_idx_sequential, mask, mask_sequential)

        h_S = self.dihedral_encode(batch, chi_flag=True, code_idx=2)

        # 无预训练
        h_V = self.single_fusion(torch.cat([h_V_spatial, h_V_sequential],dim=-1))
        E_idx = torch.cat([E_idx_spatial, E_idx_sequential], dim=-1)
        h_E = torch.cat([h_E_spatial, h_E_sequential], dim=-2)
        mask_attend = torch.cat([mask_spatial, mask_sequential], dim=-1)

        # h_V = self.single_fusion(torch.cat([h_V_spatial],dim=-1))
        # E_idx = torch.cat([E_idx_spatial], dim=-1)
        # h_E = torch.cat([h_E_spatial], dim=-2)
        # mask_attend = torch.cat([mask_spatial], dim=-1)


        h_EXV_out = self.out_layer(h_S, h_V, h_E, E_idx, mask_attend, mask)  # h_i(dec) || h_j(enc) || h_j || h_edge
        # # 中心性
        h_centrality = self.gather_centrality(batch["centrality"], E_idx, mask)
        h_EXV_out = h_EXV_out * h_centrality[:,:,None]

        return h_EXV_out

    def forward(self, batch):
        """
        Graph-conditioned sequence model
        """
        batch_wt = batch["wt"]
        batch_mt = batch["mt"]

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]

        ddg_pred = self.ddg_readout(H_mt - H_wt)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt)

        loss_mse = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        loss_dict = {
            'loss_mse': loss_mse,
        }
        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'],
        }
        return loss_dict, out_dict
