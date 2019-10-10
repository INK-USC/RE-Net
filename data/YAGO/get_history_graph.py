import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch


def load_quadruples(inPath, fileName, fileName2=None):
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
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, fileName2=None):
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
    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


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
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


graph_dict_train = {}

train_data, train_times = load_quadruples('', 'train.txt')
test_data, test_times = load_quadruples('', 'test.txt')
dev_data, dev_times = load_quadruples('', 'valid.txt')
# total_data, _ = load_quadruples('', 'train.txt', 'test.txt')

history_len = 10
num_e, num_r = get_total_number('', 'stat.txt')

s_his = [[] for _ in range(num_e)]
o_his = [[] for _ in range(num_e)]
s_his_t = [[] for _ in range(num_e)]
o_his_t = [[] for _ in range(num_e)]
s_history_data = [[] for _ in range(len(train_data))]
o_history_data = [[] for _ in range(len(train_data))]
s_history_data_t = [[] for _ in range(len(train_data))]
o_history_data_t = [[] for _ in range(len(train_data))]
e = []
r = []
latest_t = 0
s_his_cache = [[] for _ in range(num_e)]
o_his_cache = [[] for _ in range(num_e)]
s_his_cache_t = [None for _ in range(num_e)]
o_his_cache_t = [None for _ in range(num_e)]

for tim in train_times:
    print(str(tim)+'\t'+str(max(train_times)))
    data = get_data_with_t(train_data, tim)
    graph_dict_train[tim] = get_big_graph(data, num_r)

for i, train in enumerate(train_data):
    if i % 10000 == 0:
        print("train", i, len(train_data))
    # if i == 10000:
    #     break
    t = train[3]
    if latest_t != t:

        for ee in range(num_e):
            if len(s_his_cache[ee]) != 0:
                if len(s_his[ee]) >= history_len:
                    s_his[ee].pop(0)
                    s_his_t[ee].pop(0)

                s_his[ee].append(s_his_cache[ee].copy())
                s_his_t[ee].append(s_his_cache_t[ee])
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            if len(o_his_cache[ee]) != 0:
                if len(o_his[ee]) >= history_len:
                    o_his[ee].pop(0)
                    o_his_t[ee].pop(0)

                o_his[ee].append(o_his_cache[ee].copy())
                o_his_t[ee].append(o_his_cache_t[ee])
                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t
    s = train[0]
    r = train[1]
    o = train[2]
    # print(s_his[r][s])
    s_history_data[i] = s_his[s].copy()
    o_history_data[i] = o_his[o].copy()
    s_history_data_t[i] = s_his_t[s].copy()
    o_history_data_t[i] = o_his_t[o].copy()
    # print(o_history_data_g[i])

    if len(s_his_cache[s]) == 0:
        s_his_cache[s] = np.array([[r, o]])
    else:
        s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
    s_his_cache_t[s] = t

    if len(o_his_cache[o]) == 0:
        o_his_cache[o] = np.array([[r, s]])
    else:
        o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
    o_his_cache_t[o] = t

    # print(s_history_data[i], s_history_data_g[i])
    # with open('ttt.txt', 'wb') as fp:
    #     pickle.dump(s_history_data_g, fp)
    # print("save")

with open('train_graphs.txt', 'wb') as fp:
    pickle.dump(graph_dict_train, fp)

with open('train_history_sub.txt', 'wb') as fp:
    pickle.dump([s_history_data, s_history_data_t], fp)
with open('train_history_ob.txt', 'wb') as fp:
    pickle.dump([o_history_data, o_history_data_t], fp)

# print(s_history_data[0])
s_history_data_dev = [[] for _ in range(len(dev_data))]
o_history_data_dev = [[] for _ in range(len(dev_data))]
s_history_data_dev_t = [[] for _ in range(len(dev_data))]
o_history_data_dev_t = [[] for _ in range(len(dev_data))]

for i, dev in enumerate(dev_data):
    if i % 10000 == 0:
        print("valid", i, len(dev_data))
    t = dev[3]
    if latest_t != t:
        for ee in range(num_e):
            if len(s_his_cache[ee]) != 0:
                if len(s_his[ee]) >= history_len:
                    s_his[ee].pop(0)
                    s_his_t[ee].pop(0)
                s_his_t[ee].append(s_his_cache_t[ee])
                s_his[ee].append(s_his_cache[ee].copy())
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            if len(o_his_cache[ee]) != 0:
                if len(o_his[ee]) >= history_len:
                    o_his[ee].pop(0)
                    o_his_t[ee].pop(0)

                o_his_t[ee].append(o_his_cache_t[ee])
                o_his[ee].append(o_his_cache[ee].copy())

                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t
    s = dev[0]
    r = dev[1]
    o = dev[2]
    s_history_data_dev[i] = s_his[s].copy()
    o_history_data_dev[i] = o_his[o].copy()
    s_history_data_dev_t[i] = s_his_t[s].copy()
    o_history_data_dev_t[i] = o_his_t[o].copy()
    if len(s_his_cache[s]) == 0:
        s_his_cache[s] = np.array([[r, o]])
    else:
        s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
    s_his_cache_t[s] = t

    if len(o_his_cache[o]) == 0:
        o_his_cache[o] = np.array([[r, s]])
    else:
        o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
    o_his_cache_t[o] = t

    # print(o_his_cache[o])

with open('dev_history_sub.txt', 'wb') as fp:
    pickle.dump([s_history_data_dev, s_history_data_dev_t], fp)
with open('dev_history_ob.txt', 'wb') as fp:
    pickle.dump([o_history_data_dev, o_history_data_dev_t], fp)

s_history_data_test = [[] for _ in range(len(test_data))]
o_history_data_test = [[] for _ in range(len(test_data))]

s_history_data_test_t = [[] for _ in range(len(test_data))]
o_history_data_test_t = [[] for _ in range(len(test_data))]

for i, test in enumerate(test_data):
    if i % 10000 == 0:
        print("test", i, len(test_data))
    t = test[3]
    if latest_t != t:
        for ee in range(num_e):
            if len(s_his_cache[ee]) != 0:
                if len(s_his[ee]) >= history_len:
                    s_his[ee].pop(0)
                    s_his_t[ee].pop(0)
                s_his_t[ee].append(s_his_cache_t[ee])

                s_his[ee].append(s_his_cache[ee].copy())
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            if len(o_his_cache[ee]) != 0:
                if len(o_his[ee]) >= history_len:
                    o_his[ee].pop(0)
                    o_his_t[ee].pop(0)

                o_his_t[ee].append(o_his_cache_t[ee])

                o_his[ee].append(o_his_cache[ee].copy())
                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t
    s = test[0]
    r = test[1]
    o = test[2]
    s_history_data_test[i] = s_his[s].copy()
    o_history_data_test[i] = o_his[o].copy()
    s_history_data_test_t[i] = s_his_t[s].copy()
    o_history_data_test_t[i] = o_his_t[o].copy()
    if len(s_his_cache[s]) == 0:
        s_his_cache[s] = np.array([[r, o]])
    else:
        s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
    s_his_cache_t[s] = t

    if len(o_his_cache[o]) == 0:
        o_his_cache[o] = np.array([[r, s]])
    else:
        o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
    o_his_cache_t[o] = t
    # print(o_his_cache[o])

with open('test_history_sub.txt', 'wb') as fp:
    pickle.dump([s_history_data_test, s_history_data_test_t], fp)
with open('test_history_ob.txt', 'wb') as fp:
    pickle.dump([o_history_data_test, o_history_data_test_t], fp)
    # print(train)

