import os
import numpy as np


class Triple(object):
    def __init__(self, head, tail, relation):
        self.s = head
        self.o = tail
        self.r = relation
        # self.t = tim


class Quadruple(object):
    def __init__(self, head, tail, relation, tim):
        self.s = head
        self.o = tail
        self.r = relation
        self.t = tim


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, temFileName, fileName2 = None, temFileName2 = None, fileName3 = None, temFileName3 = None):
    tem = np.load(os.path.join(inPath, temFileName)).tolist()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        quadrupleTotal = 0
        times = set()
        for line in fr:
            quadrupleTotal += 1
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = tem[quadrupleTotal-1]
            # times.add(time)
            quadrupleList.append(Quadruple(head, tail, rel, time))

    trainTotal = quadrupleTotal
    if temFileName2 is not None:
        tem2 = np.load(os.path.join(inPath, temFileName2)).tolist()
    if fileName2 is not None:
        assert quadrupleTotal != 0
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                quadrupleTotal += 1
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = tem2[quadrupleTotal-trainTotal-1]
                quadrupleList.append(Quadruple(head, tail, rel, time))

    trainTestTotal = quadrupleTotal
    if temFileName3 is not None:
        tem3 = np.load(os.path.join(inPath, temFileName3)).tolist()
    if fileName3 is not None:
        assert quadrupleTotal != 0
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                quadrupleTotal += 1
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = tem3[quadrupleTotal-trainTestTotal-1]
                quadrupleList.append(Quadruple(head, tail, rel, time))

    times = list(times)
    times.sort()
    tripleDict = {}
    for quadruple in quadrupleList:
        tripleDict[(quadruple.s, quadruple.o, quadruple.r)] = True

    return quadrupleTotal, quadrupleList, tripleDict, times


def load_quadruples_TTransE(inPath, fileName, fileName2 = None, fileName3 = None):
    quadrupleList = []
    quadrupleTotal = 0

    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            quadrupleTotal += 1
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append(Quadruple(head, tail, rel, time))

    if fileName2 is not None:
        assert quadrupleTotal != 0
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                quadrupleTotal += 1
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append(Quadruple(head, tail, rel, time))

    if fileName3 is not None:
        assert quadrupleTotal != 0
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                quadrupleTotal += 1
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append(Quadruple(head, tail, rel, time))

    quadrupleDict = {}
    for quadruple in quadrupleList:
        quadrupleDict[(quadruple.s, quadruple.o, quadruple.r,  quadruple.t)] = True

    return quadrupleTotal, quadrupleList, quadrupleDict


def get_quadruple_t(quads, time):
    return [quad for quad in quads if quad.t == time]


def getTimedict(inPath, fileName):
    timedict = {}

    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            timedict[line_split[0]] = line_split[1]

    return timedict
