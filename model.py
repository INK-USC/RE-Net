import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from Aggregator import MeanAggregator, AttnAggregator, RGCNAggregator
from utils import *
import time


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
        self.encoder = nn.GRU(4 * h_dim, h_dim, batch_first=True)
        self.encoder_r = nn.GRU(3 * h_dim, h_dim, batch_first=True)

        self.preds_list_s = defaultdict(lambda: torch.zeros(self.num_k))
        self.preds_ind_s = defaultdict(lambda: torch.zeros(self.num_k))
        self.preds_list_o = defaultdict(lambda: torch.zeros(self.num_k))
        self.preds_ind_o = defaultdict(lambda: torch.zeros(self.num_k))

        self.aggregator = RGCNAggregator(h_dim, dropout, in_dim, num_rels, 100, model, seq_len)

        self.linear = nn.Linear(3 * h_dim, in_dim)
        self.linear_r = nn.Linear(2 * h_dim, num_rels)
        self.global_emb = None

        # For recording history in inference
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
        self.global_emb = None

        self.latest_time = 0

        self.criterion = nn.CrossEntropyLoss()


    """
    Prediction function in training. 
    This should be different from testing because in testing we don't use ground-truth history.
    """
    def forward(self, triplets, s_hist, o_hist, graph_dict, subject=True):
        if subject:
            rel_embeds = self.rel_embeds[:self.num_rels]
            s = triplets[:, 0]
            r = triplets[:, 1]
            o = triplets[:, 2]
            hist = s_hist
            reverse = False
        else:
            rel_embeds = self.rel_embeds[self.num_rels:]
            o = triplets[:, 0]
            r = triplets[:, 1]
            s = triplets[:, 2]
            hist = o_hist
            reverse = True

        hist_len = torch.LongTensor(list(map(len, hist[0]))).cuda()
        s_len, s_idx = hist_len.sort(0, descending=True)
        s_packed_input, s_packed_input_r = self.aggregator(hist, s, r, self.ent_embeds,
                                                        rel_embeds, graph_dict, self.global_emb,
                                                        reverse=reverse)
   
        tt, s_h = self.encoder(s_packed_input)
        s_h = s_h.squeeze()
        s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
        ob_pred = self.linear(
            self.dropout(torch.cat((self.ent_embeds[s[s_idx]], s_h, rel_embeds[r[s_idx]]), dim=1)))
        loss_sub = self.criterion(ob_pred, o[s_idx])

        ###### Relations
        tt, s_q = self.encoder_r(s_packed_input_r)
        s_q = s_q.squeeze()
        s_q = torch.cat((s_q, torch.zeros(len(s) - len(s_q), self.h_dim).cuda()), dim=0)

        ob_pred_r = self.linear_r(
            self.dropout(torch.cat((self.ent_embeds[s[s_idx]], s_q), dim=1)))
        loss_sub_r = self.criterion(ob_pred_r, r[s_idx])
        ######

        loss = loss_sub + 0.1*loss_sub_r
        return loss


    def init_history(self, triples, s_history, o_history, valid_triples, s_history_valid, o_history_valid, test_triples=None, s_history_test=None, o_history_test=None):
        s_hist = s_history[0]
        s_hist_t = s_history[1]
        o_hist = o_history[0]
        o_hist_t = o_history[1]


        self.s_hist_test = [[] for _ in range(self.in_dim)]
        self.o_hist_test = [[] for _ in range(self.in_dim)]
        self.s_hist_test_t = [[] for _ in range(self.in_dim)]
        self.o_hist_test_t = [[] for _ in range(self.in_dim)]
        self.s_his_cache = [[] for _ in range(self.in_dim)]
        self.o_his_cache = [[] for _ in range(self.in_dim)]
        self.s_his_cache_t = [None for _ in range(self.in_dim)]
        self.o_his_cache_t = [None for _ in range(self.in_dim)]

        for triple, s_his, s_his_t, o_his, o_his_t in zip(triples, s_hist, s_hist_t, o_hist, o_hist_t):
            s = triple[0]
            o = triple[2]
            last_t = triple[3]

            self.s_hist_test[s] = s_his.copy()
            self.s_hist_test_t[s] = s_his_t.copy()
            self.o_hist_test[o] = o_his.copy()
            self.o_hist_test_t[o] = o_his_t.copy()
            # print(self.o_hist_test[o])

        s_hist = s_history_valid[0]
        s_hist_t = s_history_valid[1]
        o_hist = o_history_valid[0]
        o_hist_t = o_history_valid[1]
        for triple, s_his, s_his_t, o_his, o_his_t in zip(valid_triples, s_hist, s_hist_t, o_hist, o_hist_t):
            s = triple[0]
            o = triple[2]
            t = triple[3]

            if len(s_his_t)!= 0 and s_his_t[-1] <= last_t:
                self.s_hist_test[s] = s_his.copy()
                self.s_hist_test_t[s] = s_his_t.copy()
            if len(o_his_t)!= 0 and o_his_t[-1] <= last_t:
                self.o_hist_test[o] = o_his.copy()
                self.o_hist_test_t[o] = o_his_t.copy()

        if test_triples is not None:
            s_hist = s_history_test[0]
            s_hist_t = s_history_test[1]
            o_hist = o_history_test[0]
            o_hist_t = o_history_test[1]
            for triple, s_his, s_his_t, o_his, o_his_t in zip(test_triples, s_hist, s_hist_t, o_hist, o_hist_t):
                s = triple[0]
                o = triple[2]
                t = triple[3]

                if len(s_his_t) != 0 and s_his_t[-1] <= last_t:
                    self.s_hist_test[s] = s_his.copy()
                    self.s_hist_test_t[s] = s_his_t.copy()
                if len(o_his_t) != 0 and o_his_t[-1] <= last_t:
                    self.o_hist_test[o] = o_his.copy()
                    self.o_hist_test_t[o] = o_his_t.copy()


    def pred_r_rank2(self, s, r, subject=True):
        if subject:
            s_history = []
            s_history_t = []
            s_history.append(self.s_hist_test[s[0].item()].copy())
            s_history = s_history * self.num_rels
            s_history_t.append(self.s_hist_test_t[s[0].item()].copy())
            s_history_t = s_history_t * self.num_rels
            rel_embeds = self.rel_embeds[:self.num_rels]
            reverse = False
        else:
            s_history = []
            s_history_t = []
            s_history.append(self.o_hist_test[s[0].item()].copy())
            s_history = s_history * self.num_rels
            s_history_t.append(self.o_hist_test_t[s[0].item()].copy())
            s_history_t = s_history_t * self.num_rels
            rel_embeds = self.rel_embeds[self.num_rels:]
            reverse = True
        if len(s_history[0]) == 0:
            s_h = torch.zeros(self.num_rels, self.h_dim).cuda()
            s_q = torch.zeros(self.num_rels, self.h_dim).cuda()
        else:
            s_packed_input, s_packed_input_r = self.aggregator.predict_batch((s_history, s_history_t), s, r, self.ent_embeds,
                                                   rel_embeds, self.graph_dict, self.global_emb,
                                                   reverse=reverse)
            if s_packed_input is None:
                s_h = torch.zeros(len(s), self.h_dim).cuda()
                s_q = torch.zeros(len(s), self.h_dim).cuda()
            else:
                tt, s_h = self.encoder(s_packed_input)
                s_h = s_h.squeeze()
                s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
                ###### Relations
                tt, s_q = self.encoder_r(s_packed_input_r)
                s_q = s_q.squeeze()

        ob_pred = self.linear(torch.cat((self.ent_embeds[s], s_h, rel_embeds), dim=1))
        p_o = torch.softmax(ob_pred.view(self.num_rels, self.in_dim), dim=1)
        ob_pred_r = self.linear_r(torch.cat((self.ent_embeds[s[0]], s_q[0]), dim=0))
        p_r = torch.softmax(ob_pred_r.view(-1), dim=0)
        ob_pred_rank = p_o * p_r.view(self.num_rels,1)

        return ob_pred_rank

    """
    Prediction function in testing
    """
    def predict(self, triplet, s_hist, o_hist, global_model):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]
        t = triplet[3].cpu()

        if self.latest_time != t:
            _, sub, prob_sub = global_model.predict(self.latest_time, self.graph_dict, subject=True)
            
            m = torch.distributions.categorical.Categorical(prob_sub)
            subjects = m.sample(torch.Size([self.num_k]))
            prob_subjects = prob_sub[subjects]

            s_done = set()

            for s, prob_s in zip(subjects, prob_subjects):
                if s in s_done:
                    continue
                else:
                    s_done.add(s)
                ss = torch.LongTensor([s]).repeat(self.num_rels)
                rr = torch.arange(0,self.num_rels)
                probs = prob_s * self.pred_r_rank2(ss, rr, subject=True)
                probs, indices = torch.topk(probs.view(-1), self.num_k, sorted=False)
                self.preds_list_s[s] = probs.view(-1)
                self.preds_ind_s[s] = indices.view(-1)
            s_to_id = dict()
            s_num = len(self.preds_list_s.keys())
            prob_tensor = torch.zeros(s_num * self.num_k)
            idx = 0
            for i, s in enumerate(self.preds_list_s.keys()):
                s_to_id[idx] = s
                prob_tensor[i * self.num_k: (i + 1) * self.num_k] = self.preds_list_s[s]
                idx += 1
            _, triple_candidates = torch.topk(prob_tensor, self.num_k, sorted=False)
            indices = triple_candidates // self.num_k
            for i,idx in enumerate(indices):
                s = s_to_id[idx.item()]
                num_r_num_s = self.preds_ind_s[s][triple_candidates[i] % self.num_k]
                rr = num_r_num_s // self.in_dim
                o_s = num_r_num_s % self.in_dim
                self.s_his_cache[s] = self.update_cache(self.s_his_cache[s], rr, o_s.view(-1, 1))
                self.s_his_cache_t[s] = self.latest_time.item()


            _, ob, prob_ob = global_model.predict(t, self.graph_dict, subject=False)
            prob_ob = torch.softmax(ob.view(-1), dim=0)
            m = torch.distributions.categorical.Categorical(prob_ob)
            objects = m.sample(torch.Size([self.num_k]))
            prob_objects = prob_ob[objects]

            o_done = set()
            for o, prob_o in zip(objects, prob_objects):
                if o in o_done:
                    continue
                else:
                    o_done.add(o)
                oo = torch.LongTensor([o]).repeat(self.num_rels)
                rr = torch.arange(0, self.num_rels)
                probs = prob_o * self.pred_r_rank2(oo, rr, subject=False)
                probs, indices = torch.topk(probs.view(-1), self.num_k, sorted=False)
                self.preds_list_o[o] = probs.view(-1)
                self.preds_ind_o[o] = indices.view(-1)
            o_to_id = dict()
            o_num = len(self.preds_list_o.keys())

            prob_tensor = torch.zeros(o_num * self.num_k)
            idx = 0
            for i, o in enumerate(self.preds_list_o.keys()):
                o_to_id[idx] = o
                prob_tensor[i * self.num_k: (i + 1) * self.num_k] = self.preds_list_o[o]
                idx += 1
            _, triple_candidates = torch.topk(prob_tensor, self.num_k, sorted=False)
            indices = triple_candidates // self.num_k
            for i, idx in enumerate(indices):
                o = o_to_id[idx.item()]
                num_r_num_o = self.preds_ind_o[o][triple_candidates[i] % self.num_k]
                rr = num_r_num_o // self.in_dim
                s_o = num_r_num_o % self.in_dim
                # rr = torch.tensor(rr)
                self.o_his_cache[o] = self.update_cache(self.o_his_cache[o], rr, s_o.view(-1, 1))
                self.o_his_cache_t[o] = self.latest_time.item()


            self.data = get_data(self.s_his_cache, self.o_his_cache)
            self.graph_dict[self.latest_time.item()] = get_big_graph(self.data, self.num_rels)
            global_emb_prev_t, _, _ = global_model.predict(self.latest_time, self.graph_dict, subject=True)
            self.global_emb[self.latest_time.item()] = global_emb_prev_t

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
            self.preds_list_s = defaultdict(lambda: torch.zeros(self.num_k))
            self.preds_ind_s = defaultdict(lambda: torch.zeros(self.num_k))
            self.preds_list_o = defaultdict(lambda: torch.zeros(self.num_k))
            self.preds_ind_o = defaultdict(lambda: torch.zeros(self.num_k))



        if len(s_hist[0]) == 0 or len(self.s_hist_test[s]) == 0:
            s_h = torch.zeros(self.h_dim).cuda()
        else:

            s_history = self.s_hist_test[s]
            s_history_t = self.s_hist_test_t[s]
            inp, _ = self.aggregator.predict((s_history, s_history_t), s, r, self.ent_embeds, self.rel_embeds[:self.num_rels], self.graph_dict, self.global_emb, reverse=False)
            tt, s_h = self.encoder(inp.view(1, len(s_history), 4 * self.h_dim))
            s_h = s_h.squeeze()

        if len(o_hist[0]) == 0 or len(self.o_hist_test[o]) == 0:
            o_h = torch.zeros(self.h_dim).cuda()
        else:

            o_history = self.o_hist_test[o]
            o_history_t = self.o_hist_test_t[o]
            inp, _ = self.aggregator.predict((o_history, o_history_t), o, r, self.ent_embeds, self.rel_embeds[self.num_rels:], self.graph_dict, self.global_emb, reverse=True)

            tt, o_h = self.encoder(inp.view(1, len(o_history), 4 * self.h_dim))
            o_h = o_h.squeeze()
        

        ob_pred = self.linear(torch.cat((self.ent_embeds[s], s_h, self.rel_embeds[:self.num_rels][r]), dim=0))
        sub_pred = self.linear(torch.cat((self.ent_embeds[o], o_h, self.rel_embeds[self.num_rels:][r]), dim=0))


        loss_sub = self.criterion(ob_pred.view(1, -1), o.view(-1))
        loss_ob = self.criterion(sub_pred.view(1, -1), s.view(-1))

        loss = loss_sub + loss_ob

        return loss, sub_pred, ob_pred

    def evaluate(self, triplet, s_hist, o_hist, global_model):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]

        loss, sub_pred, ob_pred = self.predict(triplet, s_hist, o_hist, global_model)
        o_label = o
        s_label = s
        ob_pred_comp1 = (ob_pred > ob_pred[o_label]).data.cpu().numpy()
        ob_pred_comp2 = (ob_pred == ob_pred[o_label]).data.cpu().numpy()
        rank_ob = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1

        sub_pred_comp1 = (sub_pred > sub_pred[s_label]).data.cpu().numpy()
        sub_pred_comp2 = (sub_pred == sub_pred[s_label]).data.cpu().numpy()
        rank_sub = np.sum(sub_pred_comp1) + ((np.sum(sub_pred_comp2) - 1.0) / 2) + 1

        return np.array([rank_sub, rank_ob]), loss


    def evaluate_filter(self, triplet, s_hist, o_hist, global_model, all_triplets):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]
        loss, sub_pred, ob_pred = self.predict(triplet, s_hist, o_hist, global_model)
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
        o_candidate = o_candidate % self.in_dim
        if len(s_his_cache) == 0:
            s_his_cache = torch.cat((r.view(-1,1),
                                     o_candidate.view(-1, 1)),
                                    dim=1)
        else:
            # print(r)
            temp = s_his_cache[torch.nonzero(s_his_cache[:, 0] == r).view(-1)]
            if len(temp) == 0:
                forward = torch.cat((r.repeat(len(o_candidate), 1), o_candidate.view(-1, 1)), dim=1)

                s_his_cache = torch.cat((s_his_cache, forward), dim=0)

            else:
                ent_list = temp[:, 1]
                tem = []
                for i in range(len(o_candidate)):
                    if o_candidate[i] not in ent_list:
                        tem.append(i)

                if len(tem) != 0:
                    forward = torch.cat((r.repeat(len(tem), 1), o_candidate[torch.LongTensor(tem)].view(-1, 1)), dim=1)

                    s_his_cache = torch.cat((s_his_cache, forward), dim=0)
        return s_his_cache

