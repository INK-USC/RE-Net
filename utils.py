import numpy as np
import os
import dgl
import torch
from collections import defaultdict


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def make_batch(a,b,c, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n], b[i:i+n], c[i:i+n]

def make_batch2(a,b,c,d,e, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n], b[i:i+n], c[i:i+n], d[i:i+n], e[i:i+n]


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for idd in uniq_v:
        g.ids[idd] = idx
        idx += 1
    return g

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def get_data(s_hist, o_hist):
    data = None
    for i, s_his in enumerate(s_hist):
        if len(s_his) != 0:
            tem = torch.cat((torch.LongTensor([i]).repeat(len(s_his), 1), torch.LongTensor(s_his.cpu())), dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)

    for i, o_his in enumerate(o_hist):
        if len(o_his) != 0:
            tem = torch.cat((torch.LongTensor(o_his[:,1].cpu()).view(-1,1), torch.LongTensor(o_his[:,0].cpu()).view(-1,1), torch.LongTensor([i]).repeat(len(o_his), 1)), dim=1)
            if data is None:
                data = tem.cpu().numpy()
            else:
                data = np.concatenate((data, tem.cpu().numpy()), axis=0)
    data = np.unique(data, axis=0)
    return data

def make_subgraph(g, nodes):
    nodes = list(nodes)
    relabeled_nodes = []

    for node in nodes:
        relabeled_nodes.append(g.ids[node])
    sub_g = g.subgraph(relabeled_nodes)

    sub_g.ndata.update({k: g.ndata[k][sub_g.parent_nid] for k in g.ndata if k != 'norm'})
    sub_g.edata.update({k: g.edata[k][sub_g.parent_eid] for k in g.edata})
    sub_g.ids = {}
    norm = comp_deg_norm(sub_g)
    sub_g.ndata['norm'] = norm.view(-1,1)

    node_id = sub_g.ndata['id'].view(-1).tolist()
    sub_g.ids.update(zip(node_id, list(range(sub_g.number_of_nodes()))))
    return sub_g


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
    g.edata.update({k: cuda(g.edata[k]) for k in g.edata})


'''
Get sorted s and r to make batch for RNN (sorted by length)
'''

def get_neighs_by_t(s_hist_sorted, s_hist_t_sorted, s_tem):
    neighs_t = defaultdict(set)
    for i, (hist, hist_t) in enumerate(zip(s_hist_sorted, s_hist_t_sorted)):
        for neighs, t in zip(hist, hist_t):
            neighs_t[t].update(neighs[:, 1].tolist())
            neighs_t[t].add(s_tem[i].item())

    return neighs_t

def get_g_list_id(neighs_t, graph_dict):
    g_id_dict = {}
    g_list = []
    idx = 0
    for tim in neighs_t.keys():
        g_id_dict[tim] = idx
        g_list.append(make_subgraph(graph_dict[tim], neighs_t[tim]))
        if idx == 0:
            g_list[idx].start_id = 0
        else:
            g_list[idx].start_id = g_list[idx - 1].start_id + g_list[idx - 1].number_of_nodes()
        idx += 1
    return g_list, g_id_dict

def get_node_ids_to_g_id(s_hist_sorted, s_hist_t_sorted, s_tem, g_list, g_id_dict):
    node_ids_graph = []
    len_s = []
    for i, hist in enumerate(s_hist_sorted):
        for j, neighs in enumerate(hist):
            len_s.append(len(neighs))
            t = s_hist_t_sorted[i][j]
            graph = g_list[g_id_dict[t]]
            node_ids_graph.append(graph.ids[s_tem[i].item()] + graph.start_id)
    return node_ids_graph, len_s

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
    embeds_split = torch.split(embeds, len_s)
    return s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split

def get_sorted_s_r_embed_rgcn(s_hist_data, s, r, ent_embeds, graph_dict, global_emb):
    s_hist = s_hist_data[0]
    s_hist_t = s_hist_data[1]
    s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_hist_sorted = []
    s_hist_t_sorted = []
    global_emb_list = []
    for i, idx in enumerate(s_idx):
        if i == num_non_zero:
            break
        s_hist_sorted.append(s_hist[idx])
        s_hist_t_sorted.append(s_hist_t[idx])
        for tt in s_hist_t[idx]:
            global_emb_list.append(global_emb[tt].view(1, ent_embeds.shape[1]).cpu())

    s_tem = s[s_idx]
    r_tem = r[s_idx]

    neighs_t = get_neighs_by_t(s_hist_sorted, s_hist_t_sorted, s_tem)

    g_list, g_id_dict = get_g_list_id(neighs_t, graph_dict)

    node_ids_graph, len_s = get_node_ids_to_g_id(s_hist_sorted, s_hist_t_sorted, s_tem, g_list, g_id_dict)

    idx = torch.cuda.current_device()
    g_list = [g.to(torch.device('cuda:'+str(idx))) for g in g_list]  
    batched_graph = dgl.batch(g_list)
    batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, ent_embeds.shape[1])

    move_dgl_to_cuda(batched_graph)
    global_emb_list = torch.cat(global_emb_list, dim=0).cuda()

    return s_len_non_zero, s_tem, r_tem, batched_graph, node_ids_graph, global_emb_list

def get_s_r_embed_rgcn(s_hist_data, s, r, ent_embeds, graph_dict, global_emb):
    s_hist = s_hist_data[0]
    s_hist_t = s_hist_data[1]
    s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()

    s_idx = torch.arange(0,len(s_hist_len))
    s_len = s_hist_len
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_hist_sorted = []
    s_hist_t_sorted = []
    global_emb_list = []
    for i, idx in enumerate(s_idx):
        if i == num_non_zero:
            break
        s_hist_sorted.append(s_hist[idx])
        s_hist_t_sorted.append(s_hist_t[idx])
        for tt in s_hist_t[idx]:
            global_emb_list.append(global_emb[tt].view(1, ent_embeds.shape[1]).cpu())

    s_tem = s[s_idx]
    r_tem = r[s_idx]

    neighs_t = get_neighs_by_t(s_hist_sorted, s_hist_t_sorted, s_tem)

    g_list, g_id_dict = get_g_list_id(neighs_t, graph_dict)

    node_ids_graph, len_s = get_node_ids_to_g_id(s_hist_sorted, s_hist_t_sorted, s_tem, g_list, g_id_dict)

    idx = torch.cuda.current_device()
    g_list = [g.to(torch.device('cuda:'+str(idx))) for g in g_list]  
    batched_graph = dgl.batch(g_list)
    batched_graph.ndata['h'] = ent_embeds[batched_graph.ndata['id']].view(-1, ent_embeds.shape[1])

    move_dgl_to_cuda(batched_graph)
    global_emb_list = torch.cat(global_emb_list, dim=0).cuda()

    return s_len_non_zero, s_tem, r_tem, batched_graph, node_ids_graph, global_emb_list


# assuming pred and soft_targets are both Variables with shape (batchsize, num_of_classes), each row of pred is predicted logits and each row of soft_targets is a discrete distribution.
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax()
    pred = pred.type('torch.DoubleTensor').cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def get_true_distribution(train_data, num_s):
    true_s = np.zeros(num_s)
    true_o = np.zeros(num_s)
    true_prob_s = None
    true_prob_o = None
    current_t = 0
    for triple in train_data:
        s = triple[0]
        o = triple[2]
        t = triple[3]
        true_s[s] += 1
        true_o[o] += 1
        if current_t != t:

            true_s = true_s / np.sum(true_s)
            true_o = true_o /np.sum(true_o)

            if true_prob_s is None:
                true_prob_s = true_s.reshape(1, num_s)
                true_prob_o = true_o.reshape(1, num_s)
            else:
                true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_s)), axis=0)
                true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_s)), axis=0)

            true_s = np.zeros(num_s)
            true_o = np.zeros(num_s)
            current_t = t


    true_prob_s = np.concatenate((true_prob_s, true_s.reshape(1, num_s)), axis=0)
    true_prob_o = np.concatenate((true_prob_o, true_o.reshape(1, num_s)), axis=0)

    return true_prob_s, true_prob_o