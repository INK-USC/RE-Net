import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from Aggregator import MeanAggregator, AttnAggregator, RGCNAggregator
from utils import *


class RENet(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, dropout=0, model=0, seq_len=10, num_k=10):
        super(RENet, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.model = model
        self.seq_len = seq_len
        self.num_k= num_k
        self.rel_embeds = nn.Parameter(torch.Tensor(2*num_rels, h_dim))
        nn.init.xavier_uniform_(self.rel_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.ent_embeds = nn.Parameter(torch.Tensor(in_dim, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.sub_encoder = nn.GRU(3 * h_dim, h_dim, batch_first=True)
        self.ob_encoder = self.sub_encoder

        if model == 0: # Attentive Aggregator
            self.aggregator_s = AttnAggregator(h_dim, dropout, seq_len)
        elif model == 1: # Mean Aggregator
            self.aggregator_s = MeanAggregator(h_dim, dropout, seq_len, gcn=False)
        elif model == 2: # Pooling Aggregator
            self.aggregator_s = MeanAggregator(h_dim, dropout, seq_len, gcn=True)
        elif model == 3: # RGCN Aggregator
            self.aggregator_s = RGCNAggregator(h_dim, dropout, in_dim, num_rels, 100, model, seq_len)
        self.aggregator_o = self.aggregator_s  

        self.linear_sub = nn.Linear(3 * h_dim, in_dim)
        self.linear_ob = self.linear_sub

        # For recording history in inference
        if model == 3:
            self.s_hist_test = None
            self.o_hist_test = None
            self.s_hist_test_t = None
            self.o_hist_test_t = None
            self.s_his_cache = None
            self.o_his_cache = None
            self.s_his_cache_t = None
            self.o_his_cache_t = None
            self.graph_dict = None
            self.data = None
        
        else:
            self.s_hist_test = None
            self.o_hist_test = None
            self.s_his_cache = None
            self.o_his_cache = None
        self.latest_time = 0

        self.criterion = nn.CrossEntropyLoss()


    """
    Prediction function in training. 
    This should be different from testing because in testing we don't use ground-truth history.
    """
    def forward(self, triplets, s_hist, o_hist, graph_dict):
        s = triplets[:, 0]
        r = triplets[:, 1]
        o = triplets[:, 2]

        if self.model == 3:
            s_hist_len = torch.LongTensor(list(map(len, s_hist[0]))).cuda()
            s_len, s_idx = s_hist_len.sort(0, descending=True)
            o_hist_len = torch.LongTensor(list(map(len, o_hist[0]))).cuda()
            o_len, o_idx = o_hist_len.sort(0, descending=True)
            s_packed_input = self.aggregator_s(s_hist, s, r, self.ent_embeds, self.rel_embeds[:self.num_rels], graph_dict, reverse=False)
            o_packed_input = self.aggregator_o(o_hist, o, r, self.ent_embeds, self.rel_embeds[self.num_rels:], graph_dict, reverse=True)
        else:
            s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
            s_len, s_idx = s_hist_len.sort(0, descending=True)
            o_hist_len = torch.LongTensor(list(map(len, o_hist))).cuda()
            o_len, o_idx = o_hist_len.sort(0, descending=True)
            s_packed_input = self.aggregator_s(s_hist, s, r, self.ent_embeds, self.rel_embeds[:self.num_rels])
            o_packed_input = self.aggregator_o(o_hist, o, r, self.ent_embeds, self.rel_embeds[self.num_rels:])


        tt, s_h = self.sub_encoder(s_packed_input)
        tt, o_h = self.ob_encoder(o_packed_input)

        s_h = s_h.squeeze()
        o_h = o_h.squeeze()

        s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
        o_h = torch.cat((o_h, torch.zeros(len(o) - len(o_h), self.h_dim).cuda()), dim=0)

        ob_pred = self.linear_sub(
            self.dropout(torch.cat((self.ent_embeds[s[s_idx]], s_h, self.rel_embeds[:self.num_rels][r[s_idx]]), dim=1)))
        sub_pred = self.linear_ob(
            self.dropout(torch.cat((self.ent_embeds[o[o_idx]], o_h, self.rel_embeds[self.num_rels:][r[o_idx]]), dim=1)))

        loss_sub = self.criterion(ob_pred, o[s_idx])
        loss_ob = self.criterion(sub_pred, s[o_idx])

        loss = loss_sub + loss_ob

        return loss, sub_pred, ob_pred, o_idx, s_idx


    def init_history(self):
        if self.model == 3:
            self.s_hist_test = [[] for _ in range(self.in_dim)]
            self.o_hist_test = [[] for _ in range(self.in_dim)]
            self.s_hist_test_t = [[] for _ in range(self.in_dim)]
            self.o_hist_test_t = [[] for _ in range(self.in_dim)]
            self.s_his_cache = [[] for _ in range(self.in_dim)]
            self.o_his_cache = [[] for _ in range(self.in_dim)]
            self.s_his_cache_t = [None for _ in range(self.in_dim)]
            self.o_his_cache_t = [None for _ in range(self.in_dim)]
        else:
            self.s_hist_test = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]
            self.o_hist_test = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]
            self.s_his_cache = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]
            self.o_his_cache = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]


    def get_loss(self, triplets, s_hist, o_hist, graph_dict):
        loss, _, _, _, _ = self.forward(triplets, s_hist, o_hist, graph_dict)
        return loss

    """
    Prediction function in testing
    """
    def predict(self, triplet, s_hist, o_hist):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]
        t = triplet[3].cpu()

        if self.model == 3:
            if self.latest_time != t:
                self.data = get_data(self.s_his_cache, self.o_his_cache)
                self.graph_dict[self.latest_time.item()] = get_big_graph(self.data, self.num_rels)
                for ee in range(self.in_dim):
                    if len(self.s_his_cache[ee]) != 0:
                        while len(self.s_hist_test[ee]) >= self.seq_len:
                            self.s_hist_test[ee].pop(0)
                            self.s_hist_test_t[ee].pop(0)
                        self.s_hist_test[ee].append(self.s_his_cache[ee].cpu().numpy().copy())
                        self.s_hist_test_t[ee].append(self.s_his_cache_t[ee])
                        self.s_his_cache[ee] = []
                        self.s_his_cache_t[ee] = None
                    if len(self.o_his_cache[ee]) != 0:
                        while len(self.o_hist_test[ee]) >= self.seq_len:
                            self.o_hist_test[ee].pop(0)
                            self.o_hist_test_t[ee].pop(0)
                        self.o_hist_test[ee].append(self.o_his_cache[ee].cpu().numpy().copy())
                        self.o_hist_test_t[ee].append(self.o_his_cache_t[ee])
                        self.o_his_cache[ee] = []
                        self.o_his_cache_t[ee] = None
                self.latest_time = t
                self.data = None
        else:
            if self.latest_time != t:
                for rr in range(self.num_rels):
                    for ee in range(self.in_dim):
                        if len(self.s_his_cache[ee][rr]) != 0:
                            if len(self.s_hist_test[ee][rr]) >= self.seq_len:
                                self.s_hist_test[ee][rr].pop(0)
                            self.s_hist_test[ee][rr].append(self.s_his_cache[ee][rr].clone())
                            self.s_his_cache[ee][rr] = []
                        if len(self.o_his_cache[ee][rr]) != 0:
                            if len(self.o_hist_test[ee][rr]) >= self.seq_len:
                                self.o_hist_test[ee][rr].pop(0)
                            self.o_hist_test[ee][rr].append(self.o_his_cache[ee][rr].clone())

                            self.o_his_cache[ee][rr] = []
                self.latest_time = t

        if self.model == 3:
            if len(s_hist[0]) == 0:
                s_h = torch.zeros(self.h_dim).cuda()
            else:
                if len(self.s_hist_test[s]) == 0:
                    self.s_hist_test[s] = s_hist[0].copy()
                    self.s_hist_test_t[s] = s_hist[1].copy()
                s_history = self.s_hist_test[s]
                s_history_t = self.s_hist_test_t[s]
                inp = self.aggregator_s.predict((s_history, s_history_t), s, r, self.ent_embeds, self.rel_embeds[:self.num_rels], self.graph_dict, reverse=False)
                tt, s_h = self.sub_rnn(inp.view(1, len(s_history), 3 * self.h_dim))
                s_h = s_h.squeeze()
            
            if len(o_hist[0]) == 0:
                o_h = torch.zeros(self.h_dim).cuda()
            else:
                if len(self.o_hist_test[o]) == 0 :
                    self.o_hist_test[o] = o_hist[0].copy()
                    self.o_hist_test_t[o] = o_hist[1].copy()
                o_history = self.o_hist_test[o]
                o_history_t = self.o_hist_test_t[o]
                inp = self.aggregator_o.predict((o_history, o_history_t), o, r, self.ent_embeds, self.rel_embeds[self.num_rels:], self.graph_dict, reverse=True)

                tt, o_h = self.ob_rnn(inp.view(1, len(o_history), 3 * self.h_dim))
                o_h = o_h.squeeze()
        
        else:
            if len(s_hist) == 0:
                s_h = torch.zeros(self.h_dim).cuda()
            else:
                if len(self.s_hist_test[s][r]) == 0:
                    self.s_hist_test[s][r] = s_hist.copy()
                s_history = self.s_hist_test[s][r]
                inp = self.aggregator_s.predict(s_history, s, r, self.ent_embeds, self.rel_embeds[:self.num_rels])
                tt, s_h = self.sub_encoder(inp.view(1, len(s_history), 3 * self.h_dim))
                s_h = s_h.squeeze()
            
            if len(o_hist) == 0:
                o_h = torch.zeros(self.h_dim).cuda()
            else:
                if len(self.o_hist_test[o][r]) == 0:
                    self.o_hist_test[o][r] = o_hist.copy()
                o_history = self.o_hist_test[o][r]
                inp = self.aggregator_o.predict(o_history, o, r, self.ent_embeds, self.rel_embeds[self.num_rels:])
                tt, o_h = self.ob_encoder(inp.view(1, len(o_history), 3 * self.h_dim))
                o_h = o_h.squeeze()

        ob_pred = self.linear_sub(torch.cat((self.ent_embeds[s], s_h, self.rel_embeds[:self.num_rels][r]), dim=0))
        sub_pred = self.linear_ob(torch.cat((self.ent_embeds[o], o_h, self.rel_embeds[self.num_rels:][r]), dim=0))

        tt, o_candidate = torch.topk(ob_pred, self.num_k)
        tt, s_candidate = torch.topk(sub_pred, self.num_k)
        if self.model == 3:
            self.s_his_cache[s] = self.update_cache(self.s_his_cache[s], r, o_candidate)
            self.o_his_cache[o] = self.update_cache(self.o_his_cache[o], r, s_candidate)
            loss_sub = self.criterion(ob_pred.view(1, -1), o.view(-1))
            loss_ob = self.criterion(sub_pred.view(1, -1), s.view(-1))
            self.s_his_cache_t[s] = t.item()
            self.o_his_cache_t[o] = t.item()
        else:
            if len(self.s_his_cache[s][r]) == 0:
                self.s_his_cache[s][r] = o_candidate
            if len(self.o_his_cache[o][r]) == 0:
                self.o_his_cache[o][r] = s_candidate

        loss_sub = self.criterion(ob_pred.view(1, -1), o.view(-1))
        loss_ob = self.criterion(sub_pred.view(1, -1), s.view(-1))

        loss = loss_sub + loss_ob

        return loss, sub_pred, ob_pred


    def evaluate(self, triplet, s_hist, o_hist):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]

        loss, sub_pred, ob_pred = self.predict(triplet, s_hist, o_hist)
        o_label = o
        s_label = s
        ob_pred_comp1 = (ob_pred > ob_pred[o_label]).data.cpu().numpy()
        ob_pred_comp2 = (ob_pred == ob_pred[o_label]).data.cpu().numpy()
        rank_ob = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1

        sub_pred_comp1 = (sub_pred > sub_pred[s_label]).data.cpu().numpy()
        sub_pred_comp2 = (sub_pred == sub_pred[s_label]).data.cpu().numpy()
        rank_sub = np.sum(sub_pred_comp1) + ((np.sum(sub_pred_comp2) - 1.0) / 2) + 1

        return np.array([rank_sub, rank_ob]), loss


    def evaluate_filter(self, triplet, s_hist, o_hist, all_triplets):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]
        loss, sub_pred, ob_pred = self.predict(triplet, s_hist, o_hist)
        o_label = o
        s_label = s
        sub_pred = F.sigmoid(sub_pred)
        ob_pred = F.sigmoid(ob_pred)

        ground = ob_pred[o].clone()

        s_id = torch.nonzero(all_triplets[:, 0] == s).view(-1)
        idx = torch.nonzero(all_triplets[s_id, 1] == r).view(-1)
        idx = s_id[idx]
        idx = all_triplets[idx, 2]
        ob_pred[idx] = 0
        ob_pred[o_label] = ground

        ob_pred_comp1 = (ob_pred > ground).data.cpu().numpy()
        ob_pred_comp2 = (ob_pred == ground).data.cpu().numpy()
        rank_ob = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1

        ground = sub_pred[s].clone()

        o_id = torch.nonzero(all_triplets[:, 2] == o).view(-1)
        idx = torch.nonzero(all_triplets[o_id, 1] == r).view(-1)
        idx = o_id[idx]
        idx = all_triplets[idx, 0]
        sub_pred[idx] = 0
        sub_pred[s_label] = ground

        sub_pred_comp1 = (sub_pred > ground).data.cpu().numpy()
        sub_pred_comp2 = (sub_pred == ground).data.cpu().numpy()
        rank_sub = np.sum(sub_pred_comp1) + ((np.sum(sub_pred_comp2) - 1.0) / 2) + 1
        return np.array([rank_sub, rank_ob]), loss

    def update_cache(self, s_his_cache, r, o_candidate):
        if len(s_his_cache) == 0:
            s_his_cache = torch.cat((r.repeat(len(o_candidate), 1), o_candidate.view(-1, 1)), dim=1)
        else:
            ent_list = s_his_cache[torch.nonzero(s_his_cache[:, 0] == r).view(-1)][:,1]
            tem = []
            for i in range(len(o_candidate)):
                if o_candidate[i] not in ent_list:
                    tem.append(i)

            if len(tem) !=0:
                forward = torch.cat((r.repeat(len(tem), 1), o_candidate[torch.LongTensor(tem)].view(-1, 1)), dim=1)

                s_his_cache = torch.cat((s_his_cache, forward), dim=0)
        return s_his_cache

