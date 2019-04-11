import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import time


class MeanAggregator(nn.Module):
    def __init__(self, h_dim, dropout, seq_len=10, gcn=False):
        super(MeanAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.gcn = gcn
        if gcn:
            self.gcn_layer = nn.Linear(h_dim, h_dim)
    
    def forward(self, s_hist, s, r, ent_embeds, rel_embeds):
        s_len_non_zero, s_tem, r_tem, embeds_stack, len_s = get_sorted_s_r_embed(s_hist, s, r, ent_embeds)

        # To get mean vector at each time
        curr = 0
        rows = []
        cols = []
        for i, leng in enumerate(len_s):
            rows.extend([i] * leng)
            cols.extend(list(range(curr,curr+leng)))
            curr += leng
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        idxes = torch.stack([rows,cols], dim=0)

        mask_tensor = torch.sparse.FloatTensor(idxes, torch.ones(len(rows)))
        mask_tensor = mask_tensor.cuda()
        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)
        embeds_mean = embeds_sum /torch.Tensor(len_s).cuda().view(-1,1)

        if self.gcn:
            embeds_mean = self.gcn_layer(embeds_mean)
            embeds_mean = F.relu(embeds_mean)
        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()
        
        # Slow!!!
        for i, embeds in enumerate(embeds_split):
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                 rel_embeds[r_tem[i]].repeat(len(embeds), 1)), dim=1)


        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)

        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                 s_len_non_zero,
                                                                 batch_first=True)

        return s_packed_input

    def predict(self, s_history, s, r, ent_embeds, rel_embeds):
        inp = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        for i, s_o in enumerate(s_history):
            tem = torch.mean(ent_embeds[s_o], dim=0)
            if self.gcn:
                tem = F.relu(self.gcn_layer(tem))
            inp[i] = torch.cat(
                (tem, ent_embeds[s], rel_embeds[r]), dim=0)
        return inp

class AttnAggregator(nn.Module):
    def __init__(self, h_dim, dropout, seq_len=10):
        super(AttnAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.attn_s = nn.Linear(3 * h_dim, h_dim)
        self.v_s = nn.Parameter(torch.Tensor(h_dim, 1))
        nn.init.xavier_uniform_(self.v_s, gain=nn.init.calculate_gain('relu'))

    def forward(self, s_hist, s, r, ent_embeds, rel_embeds):
        s_len_non_zero, s_tem, r_tem, embeds_stack, len_s = get_sorted_s_r_embed(s_hist, s, r, ent_embeds)

        ss_all = None
        rr_all = None
        curr = 0
        for i, s_l in enumerate(s_len_non_zero):
            ss = ent_embeds[s_tem[i]]
            rr = rel_embeds[r_tem[i]]
            total_num = sum(len_s[curr:curr+s_l])
            curr += s_l
            ss = ss.repeat(total_num, 1)
            rr = rr.repeat(total_num, 1)
            if i ==0 :
                ss_all = ss
                rr_all = rr
            else:
                ss_all = torch.cat([ss_all, ss], dim=0)
                rr_all = torch.cat([rr_all, rr], dim=0)

        embeds_ss_rr = torch.cat([embeds_stack, ss_all, rr_all], dim=1)
        weights = F.tanh(self.attn_s(embeds_ss_rr)) @ self.v_s
        weights_split = torch.split(weights, len_s)
        weights = torch.cat(list(map(lambda x: F.softmax(x, dim=0), weights_split)))


        curr = 0
        rows = []
        cols = []
        for i, leng in enumerate(len_s):
            rows.extend([i] * leng)
            cols.extend(list(range(curr,curr+leng)))
            curr += leng
        rows = torch.LongTensor(rows).cuda()
        cols = torch.LongTensor(cols).cuda()
        idxes = torch.stack([rows,cols], dim=0)


        mask_tensor = torch.sparse.FloatTensor(idxes, weights)
        embeds_mean = torch.sparse.mm(mask_tensor, embeds_stack)

        embeds_split2 = torch.split(embeds_mean, s_len_non_zero.tolist())
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()
        
        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)

        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                 s_len_non_zero,
                                                                 batch_first=True)

        return s_packed_input
    
    def predict(self, s_history, s, r, ent_embeds, rel_embeds):
        inp = torch.zeros(len(s_history), 3 * self.h_dim).cuda()
        for i, s_s in enumerate(s_history):
            emb_s = ent_embeds[s_s]
            ss = ent_embeds[s].repeat(len(emb_s), 1)
            rr = rel_embeds[r].repeat(len(emb_s), 1)

            emb_s_r = torch.cat((emb_s, ss, rr), dim=1)
            weights = F.softmax(F.tanh(self.attn_s(emb_s_r)) @ self.v_s, dim=0)

            inp[i] = torch.cat((torch.sum(weights * emb_s, dim=0), ent_embeds[s], rel_embeds[r]), dim=0)
        return inp


'''
Get sorted s and r to make batch for RNN (sorted by length)
'''
def get_sorted_s_r_embed(s_hist, s, r, ent_embeds):
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
    embeds = ent_embeds[torch.LongTensor(flat_s).cuda()]
    return s_len_non_zero, s_tem, r_tem, embeds, len_s