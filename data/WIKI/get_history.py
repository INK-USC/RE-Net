import numpy as np
import os
from collections import defaultdict
import pickle

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

    return np.asarray(quadrupleList), np.asarray(times)

def make_packed_rel_batches(triplets):
    init_time = triplets[0,3]
    sub_last_idx = defaultdict(lambda: 0)
    sub_last_time = defaultdict(lambda: init_time)
    ob_last_idx = defaultdict(lambda: 0)
    ob_last_time = defaultdict(lambda: init_time)
    relation_data = defaultdict(list)
    for triplet in triplets:
        s = triplet[0]
        r = triplet[1]
        o = triplet[2]
        t = triplet[3]
        relation_data[r].append([s, r, o, t])

    return relation_data

train_data, train_times = load_quadruples('','train.txt')
test_data, test_times = load_quadruples('','test.txt')
dev_data, dev_times = load_quadruples('','valid.txt')
# total_data, _ = load_quadruples('', 'train.txt', 'test.txt')

history_len = 10
num_e, num_r = get_total_number('', 'stat.txt')

s_his = [[[] for _ in range(num_e)] for _ in range(num_r)]
o_his = [[[] for _ in range(num_e)] for _ in range(num_r)]
s_history_data = [[] for _ in range(len(train_data))]
o_history_data = [[] for _ in range(len(train_data))]
e = []
r = []
latest_t = 0
s_his_cache = [[[] for _ in range(num_e)] for _ in range(num_r)]
o_his_cache = [[[] for _ in range(num_e)] for _ in range(num_r)]
for i, train in enumerate(train_data):
    if i % 10000==0:
        print("train",i, len(train_data))
    t = train[3]
    if latest_t != t:
        for rr in range(num_r):
            for ee in range(num_e):
                if len(s_his_cache[rr][ee]) != 0:
                    if len(s_his[rr][ee]) >= history_len:
                        s_his[rr][ee].pop(0)
                    s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                    s_his_cache[rr][ee]= []
                if len(o_his_cache[rr][ee]) != 0:
                    if len(o_his[rr][ee]) >=history_len:
                        o_his[rr][ee].pop(0)
                    o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_cache[rr][ee]=[]
        latest_t = t
    s = train[0]
    r = train[1]
    o = train[2]
    # print(s_his[r][s])
    s_history_data[i] = s_his[r][s].copy()
    o_history_data[i] = o_his[r][o].copy()
    s_his_cache[r][s].append(o)
    o_his_cache[r][o].append(s)
    # print(s_history_data[i])
    # print(i)
    # print("hist",s_history_data[i])
    # print(s_his_cache[r][s])
    # print(s_history_data[i])

    # print(s_his_cache[r][s])
with open('train_history_sub1.txt', 'wb') as fp:
    pickle.dump(s_history_data, fp)
with open('train_history_ob1.txt', 'wb') as fp:
    pickle.dump(o_history_data, fp)

# print(s_history_data[0])
s_history_data_dev = [[] for _ in range(len(dev_data))]
o_history_data_dev = [[] for _ in range(len(dev_data))]
    
for i, dev in enumerate(dev_data):
    if i % 10000 ==0:
        print("valid",i, len(dev_data))
    t = dev[3]
    if latest_t != t:
        for rr in range(num_r):
            for ee in range(num_e):
                if len(s_his_cache[rr][ee]) != 0:
                    if len(s_his[rr][ee]) >= history_len:
                        s_his[rr][ee].pop(0)
                    s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                    s_his_cache[rr][ee]= []
                if len(o_his_cache[rr][ee]) != 0:
                    if len(o_his[rr][ee]) >=history_len:
                        o_his[rr][ee].pop(0)
                    o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_cache[rr][ee]=[]
        latest_t = t
    s = dev[0]
    r = dev[1]
    o = dev[2]
    s_history_data_dev[i] = s_his[r][s].copy()
    o_history_data_dev[i] = o_his[r][o].copy()
    s_his_cache[r][s].append(o)
    o_his_cache[r][o].append(s)



with open('dev_history_sub1.txt', 'wb') as fp:
    pickle.dump(s_history_data_dev, fp)
with open('dev_history_ob1.txt', 'wb') as fp:
    pickle.dump(o_history_data_dev, fp)



s_history_data_test = [[] for _ in range(len(test_data))]
o_history_data_test = [[] for _ in range(len(test_data))]
    
for i, test in enumerate(test_data):
    if i % 10000 ==0:
        print("test",i, len(test_data))
    t = test[3]
    if latest_t != t:
        for rr in range(num_r):
            for ee in range(num_e):
                if len(s_his_cache[rr][ee]) != 0:
                    if len(s_his[rr][ee]) >= history_len:
                        s_his[rr][ee].pop(0)
                    s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                    s_his_cache[rr][ee]= []
                if len(o_his_cache[rr][ee]) != 0:
                    if len(o_his[rr][ee]) >=history_len:
                        o_his[rr][ee].pop(0)
                    o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                    o_his_cache[rr][ee]=[]
        latest_t = t
    s = test[0]
    r = test[1]
    o = test[2]
    s_history_data_test[i] = s_his[r][s].copy()
    o_history_data_test[i] = o_his[r][o].copy()
    s_his_cache[r][s].append(o)
    o_his_cache[r][o].append(s)



with open('test_history_sub1.txt', 'wb') as fp:
    pickle.dump(s_history_data_test, fp)
with open('test_history_ob1.txt', 'wb') as fp:
    pickle.dump(o_history_data_test, fp)
    # print(train)



