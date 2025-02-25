# coding=utf-8
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchdiffeq import odeint
from utils import *

class U2P_ODEFunc(nn.Module):
    def __init__(self,emb_dim):
        super(U2P_ODEFunc, self).__init__()
        # self.w = nn.Embedding(emb_dim,emb_dim)
        # self.w = nn.Parameter(torch.eye(emb_dim))
        # self.d = nn.Parameter(torch.zeros(emb_dim) + 1)

    def save_HG(self,HG_up,HG_pu):
        self.HG_up = HG_up
        self.HG_pu = HG_pu

    def update_e(self, emb):
        self.e = emb

    def forward(self, t, x):
        A = torch.sparse.mm(self.HG_pu,self.HG_up)
        I = torch.eye(A.shape[0]).to_sparse_coo().cuda() # 有这个效果更好
        propag_pois_embs = torch.sparse.mm(A-I, x)

        # 下面这三行是测试加上W后的
        # d = torch.clamp(self.d, min=0, max=1)
        # w = torch.mm(self.w * d, torch.t(self.w))
        # propag_pois_embs = torch.sparse.mm(propag_pois_embs, w)

        f = propag_pois_embs + self.e # 一定要加初始表征
        return f

class U2P_ODEEncoder(nn.Module):
    def __init__(self, emb_size, t = torch.tensor([0,1]),solver='euler'):
        super(U2P_ODEEncoder, self).__init__()
        self.t = t
        # self.t = torch.linspace(0, 7.0, steps=500)
        self.odefunc1hop = U2P_ODEFunc(emb_size)
        self.solver = solver

    def forward(self, all_embeddings,HG_up,HG_pu):
        t = self.t.type_as(all_embeddings)
        self.odefunc1hop.save_HG(HG_up,HG_pu) # 获取关联矩阵
        self.odefunc1hop.update_e(all_embeddings) # 这一步就是set x0
        z1 = odeint(self.odefunc1hop, all_embeddings, t, method=self.solver)[1] # 虽然这里的t是torch.tensor([0,t])，但求解器内部实际上会自适应选择多个点。
        return z1


class Geo_ODEFunc(nn.Module):
    def __init__(self,emb_dim):
        super(Geo_ODEFunc, self).__init__()
    def save_HG(self,Geo_HG):
        self.Geo_HG = Geo_HG

    def update_e(self, emb):
        self.e = emb

    def forward(self, t, x):
        self.alpha = nn.Parameter(0.8 * torch.ones(self.Geo_HG.shape)).to_sparse_coo().cuda()
        propag_pois_embs = torch.sparse.mm(self.alpha/2 * self.Geo_HG, x) - x
        f = propag_pois_embs + self.e
        return f
class Geo_ODEEncoder(nn.Module):
    def __init__(self, emb_size, t = torch.tensor([0,1]),solver='euler'):
        super(Geo_ODEEncoder, self).__init__()
        self.t = t
        self.odefunc1hop = Geo_ODEFunc(emb_size)
        self.solver = solver


    def forward(self, all_embeddings,Geo_HG):
        t = self.t.type_as(all_embeddings)
        self.odefunc1hop.save_HG(Geo_HG)
        self.odefunc1hop.update_e(all_embeddings)
        z1 = odeint(self.odefunc1hop, all_embeddings, t, method=self.solver)[1]
        return z1

class P2P_ODEFunc(nn.Module):
    def __init__(self,args):
        super(P2P_ODEFunc, self).__init__()
        self.args = args

    def save_HG(self,HG_poi_src, HG_poi_tar):
        self.HG_poi_src = HG_poi_src
        self.HG_poi_tar = HG_poi_tar

    def update_e(self, emb):
        self.e = emb

    def forward(self, t, x):
        if self.args.dataset == 'TKY' or 'SH':
            A = torch.sparse.mm(self.HG_poi_src.to_dense(), self.HG_poi_tar.to_dense())
        else:
            A = torch.sparse.mm(self.HG_poi_src,self.HG_poi_tar)

        I = torch.eye(A.shape[0]).to_sparse_coo().cuda()
        propag_pois_embs = torch.sparse.mm(A - I, x)

        f = propag_pois_embs + self.e
        return f

class P2P_ODEEncoder(nn.Module):
    def __init__(self, args, t = torch.tensor([0,1]),solver='euler'):
        super(P2P_ODEEncoder, self).__init__()
        self.t = t
        self.odefunc1hop = P2P_ODEFunc(args)
        self.solver = solver

    def forward(self, all_embeddings,HG_poi_src,HG_poi_tar):
        t = self.t.type_as(all_embeddings)
        self.odefunc1hop.save_HG(HG_poi_src,HG_poi_tar)
        self.odefunc1hop.update_e(all_embeddings)
        z1 = odeint(self.odefunc1hop, all_embeddings, t, method=self.solver)[1]
        return z1
class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer
    """
    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()

        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.3)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, HG_up, HG_pu):
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)
        return propag_pois_embs

class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network
    """

    def __init__(self, t, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.ODEEncoder = U2P_ODEEncoder(emb_dim, t)
        self.device = device
        self.dropout = dropout

    def forward(self, pois_embs, HG_up, HG_pu):
        final_pois_embs = self.ODEEncoder(pois_embs,HG_up,HG_pu)
        final_pois_embs = F.dropout(final_pois_embs, self.dropout)
        return final_pois_embs

class DirectedHyperConvLayer(nn.Module):
    """Directed hypergraph convolutional layer"""

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src

class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, t, args, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()

        self.ODEEncoder = P2P_ODEEncoder(args, t)
        self.device = device
        self.dropout = dropout

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = self.ODEEncoder(pois_embs, HG_poi_src, HG_poi_tar)
        final_pois_embs = F.dropout(final_pois_embs, self.dropout)

        return final_pois_embs

class GeoConvNetwork(nn.Module):
    def __init__(self, t, emb_dim, dropout):
        super(GeoConvNetwork, self).__init__()

        self.dropout = dropout
        self.ODEEncoder = Geo_ODEEncoder(emb_dim,t)

    def forward(self, pois_embs, geo_graph):
        output_pois_embs = self.ODEEncoder(pois_embs, geo_graph)

        return output_pois_embs


class HODE_MDP(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(HODE_MDP, self).__init__()

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature
        self.t_up = args.t1
        self.t_geo = args.t2
        self.t_p2p = args.t3
        self.cl_criterion = torch.nn.CrossEntropyLoss()
        # embedding
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)

        # embedding init
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # network
        self.mv_hconv_network = MultiViewHyperConvNetwork(torch.tensor([0, self.t_up]), args.emb_dim, 0.2, device)  # col hypergraph
        self.geo_conv_network = GeoConvNetwork(torch.tensor([0, self.t_geo]), args.emb_dim, args.dropout) # geo hypergraph
        self.di_hconv_network = DirectedHyperConvNetwork(torch.tensor([0, self.t_p2p]), args, device, args.dropout) # transitional hypergraph

        self.geo_attention = nn.Linear(args.emb_dim, 1)
        self.pref_attention = nn.Linear(args.emb_dim, 1)
        self.collab_attention = nn.Linear(args.emb_dim, 1)

        # gating before disentangled learning
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # dropout
        self.dropout = nn.Dropout(args.dropout)

    @staticmethod
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs):
        # normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        # calculate loss
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs):
        # normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        # calculate loss
        loss_cl_users = 0.0
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)

        return loss_cl_users

    def cal_loss_cl_intra(self, pois_embs):
        noise_std = 0.05
        noise = torch.randn_like(pois_embs) * noise_std
        perturb_geo_pois_embs = pois_embs + noise.cuda()
        similarity = torch.mm(pois_embs, perturb_geo_pois_embs.T)
        similarity /= 1.
        labels = torch.arange(pois_embs.size(0)).cuda()
        loss = self.cl_criterion(similarity, labels)
        return loss

    def forward(self, dataset, batch):

        geo_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],self.w_gate_geo) + self.b_gate_geo)) # 因为是weight[:-1]，所以没有取最后一个

        seq_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        hg_pois_embs = self.mv_hconv_network(col_gate_pois_embs, dataset.HG_up, dataset.HG_pu)
        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)  # [U, d]
        hg_batch_users_embs = hg_structural_users_embs[batch["user_idx"]]  # [BS, d]

        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)  # [L, d]

        # geo-aware user embeddings
        geo_structural_users_embs = torch.sparse.mm(dataset.HG_up, geo_pois_embs)
        geo_batch_users_embs = geo_structural_users_embs[batch["user_idx"]]

        # poi-poi directed hypergraph
        trans_pois_embs = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)
        # transition-aware user embeddings
        trans_structural_users_embs = torch.sparse.mm(dataset.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]  # [BS, d]

        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs)
        loss_cl_intra_hg = self.cal_loss_cl_intra(hg_pois_embs)
        loss_cl_intra_geo = self.cal_loss_cl_intra(geo_pois_embs)
        loss_cl_intra = loss_cl_intra_hg + loss_cl_intra_geo


        # normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)

        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)

        geo_scores = self.geo_attention(norm_geo_batch_users_embs)
        trans_scores = self.pref_attention(norm_trans_batch_users_embs)
        hg_scores = self.collab_attention(norm_hg_batch_users_embs)
        attention_scores = torch.cat([hg_scores, geo_scores, trans_scores], dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)
        hyper_coef = attention_weights[:, 0].unsqueeze(1)
        geo_coef = attention_weights[:, 1].unsqueeze(1)
        trans_coef = attention_weights[:, 2].unsqueeze(1)

        # final fusion for user and poi embeddings
        fusion_batch_users_embs = hyper_coef * norm_hg_batch_users_embs + geo_coef * norm_geo_batch_users_embs + trans_coef * norm_trans_batch_users_embs
        fusion_pois_embs = norm_hg_pois_embs + norm_geo_pois_embs + norm_trans_pois_embs
        seq_info = []

        for seq in batch["user_seq"]:
            valid_pois = seq[seq != self.num_pois]
            traj_pois_embs = fusion_pois_embs[valid_pois]
            average_emb = traj_pois_embs.mean(dim=0)
            seq_info.append(average_emb)
        seq_info = torch.stack(seq_info)

        prediction = (fusion_batch_users_embs + seq_info) @ fusion_pois_embs.T

        return prediction, loss_cl_user, loss_cl_poi, loss_cl_intra




