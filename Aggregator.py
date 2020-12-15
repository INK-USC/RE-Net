import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from RGCN import RGCNBlockLayer as RGCNLayer


class RGCNAggregator_global(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, model, seq_len=10, maxpool=1):
        super(RGCNAggregator_global, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.model = model
        self.maxpool = maxpool


        self.rgcn1 = RGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, num_bases,
                               activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNLayer(self.h_dim, self.h_dim, 2 * self.num_rels, num_bases,
                               activation=None, self_loop=True, dropout=dropout)


    def forward(self, t_list, ent_embeds, graph_dict, reverse):
        times = list(graph_dict.keys())
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        num_non_zero = len(torch.nonzero(t_list))
        t_list = t_list[:num_non_zero]

        for tim in t_list:
            length = int(tim // time_unit)
            if self.seq_len <= length:
                time_list.append(torch.LongTensor(times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        time_to_idx = dict()
        g_list = []
        idx = 0
        for tim in unique_t:
            time_to_idx[tim.item()] = idx
            idx += 1
            g_list.append(graph_dict[tim.item()])

        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, ent_embeds.shape[1])
        move_dgl_to_cuda(batched_graph)
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        if self.maxpool == 1:
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')

        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, self.h_dim).cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_info[time_to_idx[t.item()]]

        embed_seq_tensor = self.dropout(embed_seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        return packed_input

    def predict(self, t, ent_embeds, graph_dict, reverse):
        times = list(graph_dict.keys())

        id = 0
        for tt in times:
            if tt >= t:
                break
            id += 1

        if self.seq_len <= id:
            timess = torch.LongTensor(times[id - self.seq_len:id])
        else:
            timess = torch.LongTensor(times[:id])


        g_list = []

        for tim in timess:
            move_dgl_to_cuda(graph_dict[tim.item()])
            g_list.append(graph_dict[tim.item()])

        batched_graph = dgl.batch(g_list)
        batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, ent_embeds.shape[1])
        move_dgl_to_cuda(batched_graph)
        self.rgcn1(batched_graph, reverse)
        self.rgcn2(batched_graph, reverse)
        if self.maxpool == 1:
            global_info = dgl.max_nodes(batched_graph, 'h')
        else:
            global_info = dgl.mean_nodes(batched_graph, 'h')
        batched_graph.ndata.pop('h')

        return global_info

class RGCNAggregator(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, num_bases, model, seq_len=10):
        super(RGCNAggregator, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.model = model

        self.rgcn1 = RGCNLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases,
                               activation=F.relu, self_loop=True, dropout=dropout)
        self.rgcn2 = RGCNLayer(self.h_dim, self.h_dim, 2*self.num_rels, num_bases,
                               activation=None, self_loop=True, dropout=dropout)

    def forward(self, s_hist, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, reverse):
        length = 0
        for his in s_hist[0]:
            length += len(his)
        if length == 0:
            s_packed_input = None
        else:
            s_len_non_zero, s_tem, r_tem, g, node_ids_graph, global_emb_list = get_sorted_s_r_embed_rgcn(s_hist, s, r, ent_embeds, graph_dict, global_emb)
            if g is None:
                s_packed_input = None
            else:

                self.rgcn1(g, reverse)
                self.rgcn2(g, reverse)

                embeds_mean = g.ndata.pop('h')
                embeds_mean = embeds_mean[torch.LongTensor(node_ids_graph)]

                embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())
                global_emb_list_split = torch.split(global_emb_list, s_len_non_zero.tolist())
                s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 4 * self.h_dim).cuda()

                s_embed_seq_tensor_r = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()

                # Slow!!!
                for i, embeds in enumerate(embeds_split):
                    s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                         rel_embeds[r_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)

                    s_embed_seq_tensor_r[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)

                s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
                s_embed_seq_tensor_r = self.dropout(s_embed_seq_tensor_r)

                s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                         s_len_non_zero,
                                                                         batch_first=True)
                s_packed_input_r = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor_r,
                                                                           s_len_non_zero,
                                                                           batch_first=True)

        return s_packed_input, s_packed_input_r

    def predict_batch(self, s_hist, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, reverse):
        length = 0
        for his in s_hist[0]:
            length += len(his)
        if length == 0:
            s_packed_input = None
            s_packed_input_r = None
        else:

            s_len_non_zero, s_tem, r_tem, g, node_ids_graph, global_emb_list = get_s_r_embed_rgcn(s_hist, s, r, ent_embeds, graph_dict, global_emb)
            if g is None:
                s_packed_input = None
            else:

                self.rgcn1(g, reverse)
                self.rgcn2(g, reverse)

                embeds_mean = g.ndata.pop('h')
                embeds_mean = embeds_mean[torch.LongTensor(node_ids_graph)]

                embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())
                global_emb_list_split = torch.split(global_emb_list, s_len_non_zero.tolist())
                s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 4 * self.h_dim).cuda()

                s_embed_seq_tensor_r = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()

                # Slow!!!
                for i, embeds in enumerate(embeds_split):
                    s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1),
                         rel_embeds[r_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)

                    s_embed_seq_tensor_r[i, torch.arange(len(embeds)), :] = torch.cat(
                        (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1), global_emb_list_split[i]), dim=1)

                s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
                s_embed_seq_tensor_r = self.dropout(s_embed_seq_tensor_r)

                s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                         s_len_non_zero,
                                                                         batch_first=True)
                s_packed_input_r = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor_r,
                                                                           s_len_non_zero,
                                                                           batch_first=True)

        return s_packed_input, s_packed_input_r



    def predict(self, s_history, s, r, ent_embeds, rel_embeds, graph_dict, global_emb, reverse):
        s_hist = s_history[0]

        s_hist_t = s_history[1]
        s_len_non_zero, s_tem, r_tem, g, node_ids_graph, global_emb_list = get_s_r_embed_rgcn(([s_hist], [s_hist_t]), s.view(-1,1), r.view(-1,1), ent_embeds,
                                                                                      graph_dict, global_emb)

        self.rgcn1(g, reverse)
        self.rgcn2(g, reverse)
        embeds_mean = g.ndata.pop('h')
        embeds = embeds_mean[torch.LongTensor(node_ids_graph)]

        inp = torch.zeros(len(s_hist), 4 * self.h_dim).cuda()
        inp[torch.arange(len(embeds)), :] = torch.cat(
            (embeds, ent_embeds[s].repeat(len(embeds), 1), rel_embeds[r].repeat(len(embeds), 1), global_emb_list), dim=1)

        inp_r = torch.zeros(len(s_hist), 3 * self.h_dim).cuda()
        inp_r[torch.arange(len(embeds)), :] = torch.cat((embeds, ent_embeds[s].repeat(len(embeds), 1), global_emb_list), dim=1)

        return inp, inp_r

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
        s_len_non_zero, s_tem, r_tem, embeds_stack, len_s, embeds_split = get_sorted_s_r_embed(s_hist, s, r, ent_embeds)

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
        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 2 * self.h_dim).cuda()
        
        # Slow!!!
        for i, embeds in enumerate(embeds_split):
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, ent_embeds[s_tem[i]].repeat(len(embeds), 1)), dim=1)


        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)

        s_packed_input = torch.nn.utils.rnn.pack_padded_sequence(s_embed_seq_tensor,
                                                                 s_len_non_zero,
                                                                 batch_first=True)

        return s_packed_input

    def predict(self, s_history, s, r, ent_embeds, rel_embeds):
        inp = torch.zeros(len(s_history), 2 * self.h_dim).cuda()
        for i, s_o in enumerate(s_history):
            tem = torch.mean(ent_embeds[s_o], dim=0)
            if self.gcn:
                tem = F.relu(self.gcn_layer(tem))
            inp[i] = torch.cat(
                (tem, ent_embeds[s]), dim=0)
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
        s_len_non_zero, s_tem, r_tem, embeds_stack, len_s, embeds_split = get_sorted_s_r_embed(s_hist, s, r, ent_embeds)

        s_embed_seq_tensor = torch.zeros(len(s_len_non_zero), self.seq_len, 3 * self.h_dim).cuda()

        curr = 0
        for i, s_l in enumerate(s_len_non_zero):
            # Make a batch, get first elements from all sequences, and get second elements from all sequences
            em = embeds_split[curr:curr + s_l]
            len_s = list(map(len, em))
            curr += s_l

            em_cat = torch.cat(em, dim=0)
            ss = ent_embeds[s_tem[i]]
            rr = rel_embeds[r_tem[i]]
            ss = ss.repeat(len(em_cat), 1)
            rr = rr.repeat(len(em_cat), 1)
            em_s_r = torch.cat((em_cat, ss, rr), dim=1)
            weights = F.tanh(self.attn_s(em_s_r)) @ self.v_s
            weights_split = torch.split(weights, len_s)
            weights_cat = list(map(lambda x: F.softmax(x, dim=0), weights_split))
            embeds = torch.stack(list(map(lambda x, y: torch.sum(x * y, dim=0), weights_cat, em)))
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
        for i, s_s in enumerate(s_history):
            emb_s = ent_embeds[s_s]
            ss = ent_embeds[s].repeat(len(emb_s), 1)
            rr = rel_embeds[r].repeat(len(emb_s), 1)

            emb_s_r = torch.cat((emb_s, ss, rr), dim=1)
            weights = F.softmax(F.tanh(self.attn_s(emb_s_r)) @ self.v_s, dim=0)

            inp[i] = torch.cat((torch.sum(weights * emb_s, dim=0), ent_embeds[s], rel_embeds[r]), dim=0)
        return inp


