"""
-*- coding: utf-8 -*-


@Time : 2024/3/9 21:50
@File : MA3N.py
@function :
"""
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from utils import distance_correlation


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):

    row, col = edge_index[0], edge_index[1]

    deg = torch.zeros(num_nodes,device=edge_weight.device)

    deg = deg.index_add_(0,row,edge_weight)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm


def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


class MA3N(torch.nn.Module):
    def __init__(self, num_user, num_item, n_factors, edge_index, user_item_dict, v_feat, t_feat, dim_E, reg_weight,
                 n_layers, aggr_mode, ssl_temp, ssl_alpha, cor_alpha, device):
        super(MA3, self).__init__()
        self.result = None
        self.num_user = num_user
        self.num_item = num_item
        self.n_factors = n_factors
        self.edge_index = edge_index
        self.user_item_dict = user_item_dict
        self.dim_E = dim_E
        self.n_layers = 1
        self.n_ui_layers = 2
        self.ssl_temp = ssl_temp
        self.ssl_alpha = ssl_alpha
        self.cor_alpha = cor_alpha
        self.device = device
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.reg_weight = reg_weight
        self.aggr_mode = aggr_mode

       
        self.sparse = True

        
        self.knn_k = 10

        adjusted_item_ids = edge_index[:, 1] - self.num_user
      
        self.interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),  
                                                 (edge_index[:, 0], adjusted_item_ids)),  
                                                shape=(self.num_user, self.num_item), dtype=np.float32)


        self.user_embedding = nn.Embedding(self.num_user, self.dim_E)
        self.item_embedding = nn.Embedding(self.num_item, self.dim_E)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)


        self.norm_adj = self.get_adj_mat()
    
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)

        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

     
        self.image_embedding = nn.Embedding.from_pretrained(v_feat, freeze=False)
      
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                               norm_type='sym')
        self.image_original_adj = image_adj.to(self.device)

        self.text_embedding = nn.Embedding.from_pretrained(t_feat, freeze=False)
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        self.text_original_adj = text_adj.to(self.device)

      
        self.image_trs = nn.Linear(v_feat.shape[1], self.dim_E)
        self.text_trs = nn.Linear(t_feat.shape[1], self.dim_E)

        self.softmax = nn.Softmax(dim=-1)

  
        self.query_common = nn.Sequential(
            nn.Linear(self.dim_E // self.n_factors, self.dim_E // self.n_factors),
            nn.Tanh(),
            nn.Linear(self.dim_E // self.n_factors, 1, bias=False)
        )

   
        self.gate_v = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.dim_E, self.dim_E),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.dim_E // self.n_factors, self.dim_E // self.n_factors),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.dim_E // self.n_factors, self.dim_E // self.n_factors),
            nn.Sigmoid()
        )

    def get_adj_mat(self):
  
        adj_mat = sp.dok_matrix((self.num_user + self.num_item, self.num_user + self.num_item), dtype=np.float32)
        adj_mat = adj_mat.tolil()
       
        R = self.interaction_matrix.tolil()

     
        adj_mat[:self.num_user, self.num_user:] = R
        adj_mat[self.num_user:, :self.num_user] = R.T
      
        adj_mat = adj_mat.todok()

      
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            
            rowsum[rowsum == 0.] = 1e-16
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)

            return norm_adj.tocoo()

        
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.num_user, self.num_user:]
        
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
       
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

        
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))


        values = torch.from_numpy(sparse_mx.data)

   
        shape = torch.Size(sparse_mx.shape)


        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        # User-Guided Purifier
        image_item_embeds = torch.multiply(self.item_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_embedding.weight, self.gate_t(text_feats))

        # User-Item View
        item_embeds = self.item_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # print('content_embeds:',content_embeds.size())

        user_embeds, item_embeds = torch.split(content_embeds, [self.num_user, self.num_item], dim=0)

        # Item-Item View
        # if self.sparse:
        #     for i in range(self.n_layers):
        #         image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        #         text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        # else:
        #     for i in range(self.n_layers):
        #         image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        #         text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        # print('img_user_embeds:',image_user_embeds.size())
        # print('text_user_embeds:',text_user_embeds.size())
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # print("image_embeds:", image_embeds.size())
        # print("text_embeds:", text_embeds.size())

        # Disentangled Representation
        user = torch.chunk(user_embeds, self.n_factors, dim=1)
        item = torch.chunk(item_embeds, self.n_factors, dim=1)
        text = torch.chunk(image_embeds, self.n_factors, dim=1)
        image = torch.chunk(text_embeds, self.n_factors, dim=1)

        # User-Aware Fuser
        sqrt_d = torch.sqrt(torch.tensor(self.dim_E // self.n_factors))
        all_embeds_list = []
        for i in range(0, self.n_factors):
            users_items = torch.cat([user[i],item[i]], dim=0)
            attn_text = torch.exp((users_items * text[i]) / sqrt_d)
            attn_visual = torch.exp((users_items * image[i]) / sqrt_d)

            common_embeds = attn_visual * image[i] + attn_text * text[i]
            sep_image_embeds = image[i] - common_embeds
            sep_text_embeds = text[i] - common_embeds

            image_prefer = self.gate_image_prefer(users_items)
            text_prefer = self.gate_text_prefer(users_items)
            sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
            sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
            side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

  
            all_embeds_i = users_items[i] + side_embeds

            all_embeds_list.append(all_embeds_i)

        all_embeds = torch.cat(all_embeds_list, dim=1)

        # print('all:',all_embeds.size())
        # print('side:',side_embeds.size())
        self.result = all_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.num_user, self.num_item], dim=0)

        return all_embeddings_users, all_embeddings_items, user, item, text, image

    def bpr_loss(self, users, pos_items, neg_items, u_g, i_g):
      
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]

     
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)

        # 计算 BPR 损失
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5))

        return loss

    def regularization_loss(self, users, pos_items, neg_items, u_g, i_g):
      
        user_embeddings = u_g[users]
        pos_item_embeddings = i_g[pos_items]
        neg_item_embeddings = i_g[neg_items]
        reg_loss = self.reg_weight * (
                torch.mean(user_embeddings ** 2) + torch.mean(pos_item_embeddings ** 2) + torch.mean(
            neg_item_embeddings ** 2))

        return reg_loss

    def ssl_loss(self, users, pos_items, user_embed, item_embed, text_embed, image_embed):
        ssl_loss = torch.zeros(1).to(self.device)
        for i in range(0, self.n_factors):
            image_embeds_users, image_embeds_items = torch.split(image_embed[i], [self.num_user, self.num_item], dim=0)
            text_embeds_users, text_embeds_items = torch.split(text_embed[i], [self.num_user, self.num_item], dim=0)
            user_embeds = user_embed[i]
            item_embeds = item_embed[i]

            ssl_loss += self.InfoNCE(image_embeds_items[pos_items], text_embeds_items[pos_items]) + self.InfoNCE(
                image_embeds_users[users], text_embeds_users[users])

            ssl_loss += self.InfoNCE(item_embeds[pos_items], text_embeds_items[pos_items]) + self.InfoNCE(
                user_embeds[users], text_embeds_users[users])

            ssl_loss += self.InfoNCE(item_embeds[pos_items], image_embeds_items[pos_items]) + self.InfoNCE(
                user_embeds[users], image_embeds_users[users])

        ssl_loss /= ((self.n_factors + 1) * self.n_factors / 2)

        return ssl_loss

    def InfoNCE(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.ssl_temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def cor_loss(self,users, pos_items, user, item, text_embed, image_embed):

        cor_loss = torch.zeros(1).to(self.device)
        for i in range(0, self.n_factors - 1):
            user_i = user[i]
            user_j = user[i + 1]

            user_tensor_i = user_i[users]
            user_tensor_j = user_j[users]

            cor_loss += distance_correlation(user_tensor_i, user_tensor_j, self.device)

            item_i = item[i]
            item_j = item[i + 1]

            item_tensor_i = item_i[pos_items]
            item_tensor_j = item_j[pos_items]

            cor_loss += distance_correlation(item_tensor_i, item_tensor_j, self.device)

            text_embed_i = text_embed[i]
            text_embed_j = text_embed[i + 1]

            text_u_i = text_embed_i[users]
            text_u_j = text_embed_j[users]
            text_i_i = text_embed_i[self.num_user + pos_items]
            text_i_j = text_embed_j[self.num_user + pos_items]

            t_embeddings_i = torch.cat([text_u_i, text_i_i], dim=0)
            t_embeddings_j = torch.cat([text_u_j, text_i_j], dim=0)

            cor_loss += distance_correlation(t_embeddings_i, t_embeddings_j, self.device)

            image_embed_i = image_embed[i]
            image_embed_j = image_embed[i + 1]

            image_u_i = image_embed_i[users]
            image_u_j = image_embed_j[users]
            image_i_i = image_embed_i[self.num_user + pos_items]
            image_i_j = image_embed_j[self.num_user + pos_items]

            v_embeddings_i = torch.cat([image_u_i, image_i_i], dim=0)
            v_embeddings_j = torch.cat([image_u_j, image_i_j], dim=0)

            cor_loss += distance_correlation(v_embeddings_i, v_embeddings_j, self.device)

        cor_loss /= ((self.n_factors + 1) * self.n_factors / 2)

        return cor_loss

    def loss(self, users, pos_items, neg_items):
        pos_items = pos_items - self.num_user
        neg_items = neg_items - self.num_user
        users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

        ua_embeddings, ia_embeddings, user_embed, item_embed, text_embed, image_embed = self.forward()

        bpr_loss = self.bpr_loss(users, pos_items, neg_items, ua_embeddings, ia_embeddings)
        reg_loss = self.regularization_loss(users, pos_items, neg_items, ua_embeddings, ia_embeddings)
        ssl_loss = self.ssl_loss(users, pos_items ,user_embed, item_embed, text_embed, image_embed)
        cor_loss = self.cor_loss(users, pos_items, user_embed, item_embed, text_embed, image_embed)

        total_loss = bpr_loss + self.ssl_alpha * ssl_loss + reg_loss + self.cor_alpha * cor_loss

        # print('cor_loss:',cor_loss,'alpha:',self.cor_alpha)
        # print('ssl_loss:', ssl_loss, 'alpha:', self.ssl_alpha)
        # print('bpr_loss:', bpr_loss, 'reg_loss:', reg_loss)

        return total_loss

    def gene_ranklist(self, topk=50):
      
        user_tensor = self.result[:self.num_user].cpu()
        item_tensor = self.result[self.num_user:self.num_user + self.num_item].cpu()

      
        all_index_of_rank_list = torch.LongTensor([])

      
        score_matrix = torch.matmul(user_tensor, item_tensor.t())

      
        for row, col in self.user_item_dict.items():
            col = torch.LongTensor(list(col)) - self.num_user
            score_matrix[row][col] = 1e-6

        
        _, index_of_rank_list_train = torch.topk(score_matrix, topk)
        
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list_train.cpu() + self.num_user),
            dim=0)

      
        return all_index_of_rank_list
