import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
import time


class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, dropout=0, model=0, seq_len=10, rnn_layers=1, num_k=10):
        super(LinkPredict, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.model = model
        self.seq_len = seq_len
        self.num_k = num_k
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.rel_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.ent_embeds = nn.Parameter(torch.Tensor(in_dim, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        # Attentive aggregator
        if model == 0:
            self.sub_rnn = nn.GRU(3 * h_dim, h_dim, batch_first=True)
            self.ob_rnn = nn.GRU(3 * h_dim, h_dim, batch_first=True)

            self.attn_s = nn.Linear(3 * h_dim, h_dim)
            self.attn_o = nn.Linear(3 * h_dim, h_dim)
            self.v_s = nn.Parameter(torch.Tensor(h_dim, 1))
            nn.init.xavier_uniform_(self.v_s, gain=nn.init.calculate_gain('relu'))
            self.v_o = nn.Parameter(torch.Tensor(h_dim, 1))
            nn.init.xavier_uniform_(self.v_o, gain=nn.init.calculate_gain('relu'))



        elif model == 1 or model == 2:
            self.sub_rnn = nn.GRU(3 * h_dim, h_dim, batch_first=True)
            self.ob_rnn = nn.GRU(3 * h_dim, h_dim, batch_first=True)

            if model == 2:  # gcn aggregator
                self.gcn_layer_sub = nn.Linear(h_dim, h_dim)
                self.gcn_layer_ob = nn.Linear(h_dim, h_dim)

        self.linear_sub = nn.Linear(3 * h_dim, in_dim)
        self.linear_ob = nn.Linear(3 * h_dim, in_dim)

        # For recording history in inference
        self.s_hist_test = [[[] for _ in range(num_rels)] for _ in range(in_dim)]
        self.o_hist_test = [[[] for _ in range(num_rels)] for _ in range(in_dim)]
        self.s_his_cache = [[[] for _ in range(num_rels)] for _ in range(in_dim)]
        self.o_his_cache = [[[] for _ in range(num_rels)] for _ in range(in_dim)]
        self.latest_time = 0

        self.criterion = nn.CrossEntropyLoss()

    '''
    Get sorted s and r to make batch for RNN (sorted by length)
    '''
    def get_sorted_s_r_embed(self, s_hist, s, r):
        s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
        s_len, s_idx = s_hist_len.sort(0, descending=True)
        num_non_zero = len(torch.nonzero(s_len))
        s_len_non_zero = s_len[:num_non_zero]

        s_hist_sorted = []
        for idx in s_idx:
            s_hist_sorted.append(s_hist[idx.item()])
        flat_s = []
        len_s = []
        s_hist_sorted = s_hist_sorted[:num_non_zero]
        for hist in s_hist_sorted:
            for neighs in hist:
                len_s.append(len(neighs))
                for neigh in neighs:
                    flat_s.append(neigh)
        s_tem = s[s_idx]
        r_tem = r[s_idx]
        embeds = self.ent_embeds[torch.LongTensor(flat_s).cuda()]
        embeds_split = torch.split(embeds, len_s)
        return s_len_non_zero, s_tem, r_tem, embeds_split

    '''
    Attention aggregator
    '''
    def get_input_model_0(self, s_hist, s, r, attn_s, v_s):
        s_len_non_zero, s_tem, r_tem, embeds_split = self.get_sorted_s_r_embed(s_hist, s, r)
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()

        curr = 0
        for i, s_l in enumerate(s_len_non_zero):
            # Make a batch, get first elements from all sequences, and get second elements from all sequences
            em = embeds_split[curr:curr + s_l]
            len_s = list(map(len, em))
            curr += s_l

            em_cat = torch.cat(em, dim=0)
            ss = self.ent_embeds[s_tem[i]]
            rr = self.ent_embeds[r_tem[i]]
            ss = ss.repeat(len(em_cat), 1)
            rr = rr.repeat(len(em_cat), 1)
            em_s_r = torch.cat((em_cat, ss, rr), dim=1)
            weights = F.tanh(attn_s(em_s_r)) @ v_s
            weights_split = torch.split(weights, len_s)
            weights_cat = list(map(lambda x: F.softmax(x, dim=0), weights_split))
            embeds = torch.stack(list(map(lambda x, y: torch.sum(x * y, dim=0), weights_cat, em)))
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, self.ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                 self.rel_embeds[r_tem[i]].repeat(len(embeds), 1)), dim=1)

        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)

        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                 s_len_non_zero,
                                                                 batch_first=True)

        return s_packed_input

    '''
    Mean aggregator and GCN aggregator
    '''
    def get_input_model_1_2(self, s_hist, s, r, gcn_layer=None):
        s_len_non_zero, s_tem, r_tem, embeds_split = self.get_sorted_s_r_embed(s_hist, s, r)
        # To get mean vector at each time
        embeds_mean = torch.stack(list(map(lambda x: torch.mean(x, dim=0), embeds_split)))
        if self.model == 2:
            embeds_mean = gcn_layer(embeds_mean)
            embeds_mean = F.relu(embeds_mean)
        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()
        for i, embeds in enumerate(embeds_split):
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, self.ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                 self.rel_embeds[r_tem[i]].repeat(len(embeds), 1)), dim=1)

        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)

        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                 s_len_non_zero,
                                                                 batch_first=True)
        return s_packed_input


    """
    Prediction function in training. 
    This should be different from testing because in testing we don't use ground-truth history.
    """
    def forward(self, triplets, s_hist, o_hist):
        s = triplets[:, 0]
        r = triplets[:, 1]
        o = triplets[:, 2]
        s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
        s_len, s_idx = s_hist_len.sort(0, descending=True)
        o_hist_len = torch.LongTensor(list(map(len, o_hist))).cuda()
        o_len, o_idx = o_hist_len.sort(0, descending=True)

        if self.model == 0:
            s_packed_input = self.get_input_model_0(s_hist, s, r, self.attn_s, self.v_s)
            o_packed_input = self.get_input_model_0(o_hist, o, r, self.attn_o, self.v_o)
        elif self.model ==1:
            s_packed_input = self.get_input_model_1_2(s_hist, s, r)
            o_packed_input = self.get_input_model_1_2(o_hist, o, r)
        elif self.model ==2:
            s_packed_input = self.get_input_model_1_2(s_hist, s, r, self.gcn_layer_sub)
            o_packed_input = self.get_input_model_1_2(o_hist, o, r, self.gcn_layer_ob)


        out_s, s_h = self.sub_rnn(s_packed_input)
        out_o, o_h = self.ob_rnn(o_packed_input)

        s_h = s_h.squeeze()
        o_h = o_h.squeeze()

        # print(s_h.shape)

        s_h = torch.cat((s_h, torch.zeros(len(s) - len(s_h), self.h_dim).cuda()), dim=0)
        o_h = torch.cat((o_h, torch.zeros(len(o) - len(o_h), self.h_dim).cuda()), dim=0)

        ob_pred = self.linear_sub(
            self.dropout(torch.cat((self.ent_embeds[s[s_idx]], s_h, self.rel_embeds[r[s_idx]]), dim=1)))
        sub_pred = self.linear_ob(
            self.dropout(torch.cat((self.ent_embeds[o[o_idx]], o_h, self.rel_embeds[r[o_idx]]), dim=1)))

        loss_sub = self.criterion(ob_pred, o[s_idx])
        loss_ob = self.criterion(sub_pred, s[o_idx])

        loss = loss_sub + loss_ob

        return loss, sub_pred, ob_pred, o_idx, s_idx


    def init_history(self):
        self.s_hist_test = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]
        self.o_hist_test = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]
        self.s_his_cache = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]
        self.o_his_cache = [[[] for _ in range(self.num_rels)] for _ in range(self.in_dim)]


    def get_loss(self, triplets, s_hist, o_hist):
        loss, _, _, _, _ = self.forward(triplets, s_hist, o_hist)
        return loss

    def get_input_model_0_pred(self, s_history, s, r, attn_s, v_s):
        inp = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        for i, s_s in enumerate(s_history):
            emb_s = self.ent_embeds[s_s]
            ss = self.ent_embeds[s].repeat(len(emb_s), 1)
            rr = self.rel_embeds[r].repeat(len(emb_s), 1)

            emb_s_r = torch.cat((emb_s, ss, rr), dim=1)
            weights = F.softmax(F.tanh(attn_s(emb_s_r)) @ v_s, dim=0)

            inp[i] = torch.cat((torch.sum(weights * emb_s, dim=0), self.ent_embeds[s], self.rel_embeds[r]), dim=0)
        return inp

    def get_input_model_1_2_pred(self, s_history, s, r, gcn_layer=None):
        inp = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        for i, s_o in enumerate(s_history):
            tem = torch.mean(self.ent_embeds[s_o], dim=0)
            if self.model == 2:
                tem = F.relu(gcn_layer(tem))
            inp[i] = torch.cat(
                (tem, self.ent_embeds[s], self.rel_embeds[r]), dim=0)
        return inp
    """
    Prediction function in testing
    """
    def predict(self, triplet, s_hist, o_hist):
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]
        t = triplet[3].cpu()

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

        # If there is no history
        if len(s_hist) == 0 and len(o_hist) == 0:
            s_h = torch.zeros(self.h_dim).cuda()
            o_h = torch.zeros(self.h_dim).cuda()

        # If o has history
        elif len(s_hist) == 0 and len(o_hist) != 0:
            s_h = torch.zeros(self.h_dim).cuda()
            if len(self.o_hist_test[o][r]) == 0:
                self.o_hist_test[o][r] = o_hist.copy()
            o_history = self.o_hist_test[o][r]

            if self.model == 0:
                inp = self.get_input_model_0_pred(o_history, o, r, self.attn_o, self.v_o)
            elif self.model == 1:
                inp = self.get_input_model_1_2_pred(o_history, o, r)
            elif self.model == 2:
                inp = self.get_input_model_1_2_pred(o_history, o, r, self.gcn_layer_ob)

            tt, o_h = self.ob_rnn(inp.view(1, len(o_history), 3 * self.h_dim))
            o_h = o_h.squeeze()

        elif len(s_hist) != 0 and len(o_hist) == 0:
            o_h = torch.zeros(self.h_dim).cuda()
            if len(self.s_hist_test[s][r]) == 0:
                self.s_hist_test[s][r] = s_hist.copy()
            s_history = self.s_hist_test[s][r]

            if self.model == 0:
                inp = self.get_input_model_0_pred(s_history, s, r, self.attn_s, self.v_s)
            elif self.model == 1:
                inp = self.get_input_model_1_2_pred(s_history, s, r)
            elif self.model == 2:
                inp = self.get_input_model_1_2_pred(s_history, s, r, self.gcn_layer_sub)


            tt, s_h = self.sub_rnn(inp.view(1, len(s_history), 3 * self.h_dim))
            s_h = s_h.squeeze()

        elif len(s_hist) != 0 and len(o_hist) != 0:

            if len(self.o_hist_test[o][r]) == 0:
                self.o_hist_test[o][r] = o_hist.copy()
            o_history = self.o_hist_test[o][r]

            if self.model == 0:
                inp = self.get_input_model_0_pred(o_history, o, r, self.attn_o, self.v_o)
            elif self.model == 1:
                inp = self.get_input_model_1_2_pred(o_history, o, r)
            elif self.model == 2:
                inp = self.get_input_model_1_2_pred(o_history, o, r, self.gcn_layer_ob)

            tt, o_h = self.ob_rnn(inp.view(1, len(o_history), 3 * self.h_dim))
            o_h = o_h.squeeze()

            if len(self.s_hist_test[s][r]) == 0:
                self.s_hist_test[s][r] = s_hist.copy()
            s_history = self.s_hist_test[s][r]

            if self.model == 0:
                inp = self.get_input_model_0_pred(s_history, s, r, self.attn_s, self.v_s)
            elif self.model == 1:
                inp = self.get_input_model_1_2_pred(s_history, s, r)
            elif self.model == 2:
                inp = self.get_input_model_1_2_pred(s_history, s, r, self.gcn_layer_sub)

            tt, s_h = self.sub_rnn(inp.view(1, len(s_history), 3 * self.h_dim))
            s_h = s_h.squeeze()

        ob_pred = self.linear_sub(torch.cat((self.ent_embeds[s], s_h, self.rel_embeds[r]), dim=0))
        sub_pred = self.linear_ob(torch.cat((self.ent_embeds[o], o_h, self.rel_embeds[r]), dim=0))

        tt, o_candidate = torch.topk(ob_pred, self.num_k)
        tt, s_candidate = torch.topk(sub_pred, self.num_k)
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

