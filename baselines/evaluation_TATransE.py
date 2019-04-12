import numpy as np
import torch
import torch.autograd as autograd

from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from data import *
from eval_lib import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor


# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, array, tripleDict):
    wrongAnswer = 0
    for num in array:
        if num == tail:
            return wrongAnswer
        elif (head, num, rel) in tripleDict:
            continue
        else:
            wrongAnswer += 1
    return wrongAnswer

# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(head, tail, rel, array, tripleDict):
    wrongAnswer = 0
    for num in array:
        if num == head:
            return wrongAnswer
        elif (num, tail, rel) in tripleDict:
            continue
        else:
            wrongAnswer += 1
    return wrongAnswer


def evaluation_helper(testList, tripleDict, model, ent_embeddings, L1_flag, filter, head=0):
    # embeddings are numpy likre
    headList, tailList, relList, timeList = getFourElements(testList)
    h_e = ent_embeddings[headList]
    t_e = ent_embeddings[tailList]

    test_r_batch = autograd.Variable(longTensor(relList))
    test_time_batch = autograd.Variable(longTensor(timeList))

    rseq_e = model.get_rseq(test_r_batch, test_time_batch).data.cpu().numpy()

    c_t_e = h_e + rseq_e
    c_h_e = t_e - rseq_e
    if L1_flag == True:
        dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
    else:
        dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')

    rankArrayTail = np.argsort(dist, axis=1)
    if filter == False:
        rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
    else:
        rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict)
                        for elem in zip(headList, tailList, relList, rankArrayTail)]

    isHit1ListTail = [x for x in rankListTail if x < 1]
    isHit3ListTail = [x for x in rankListTail if x < 3]
    isHit10ListTail = [x for x in rankListTail if x < 10]

    if L1_flag == True:
        dist = pairwise_distances(c_h_e, ent_embeddings, metric='manhattan')
    else:
        dist = pairwise_distances(c_h_e, ent_embeddings, metric='euclidean')

    rankArrayHead = np.argsort(dist, axis=1)
    if filter == False:
        rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
    else:
        rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict)
                        for elem in zip(headList, tailList, relList, rankArrayHead)]

    re_rankListHead = [1.0/(x+1) for x in rankListHead]
    re_rankListTail = [1.0/(x+1) for x in rankListTail]

    isHit1ListHead = [x for x in rankListHead if x < 1]
    isHit3ListHead = [x for x in rankListHead if x < 3]
    isHit10ListHead = [x for x in rankListHead if x < 10]

    totalRank = sum(rankListTail) + sum(rankListHead)
    totalReRank = sum(re_rankListHead) + sum(re_rankListTail)
    hit1Count = len(isHit1ListTail) + len(isHit1ListHead)
    hit3Count = len(isHit3ListTail) + len(isHit3ListHead)
    hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
    tripleCount = len(rankListTail) + len(rankListHead)

    return hit1Count, hit3Count, hit10Count, totalRank, totalReRank, tripleCount


def process_data(testList, tripleDict, model, ent_embeddings, L1_flag, filter, L, head):
    hit1Count, hit3Count, hit10Count, totalRank, totalReRank, tripleCount = evaluation_helper(testList, tripleDict, model, ent_embeddings, L1_flag, filter, head)

    L.append((hit1Count, hit3Count, hit10Count, totalRank, totalReRank, tripleCount))


def evaluation(testList, tripleDict, model, ent_embeddings, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, model, ent_embeddings, L1_flag, filter, L, head)

    resultList = list(L)

    # what is head?
    if head == 1 or head == 2:
        hit1 = sum([elem[0] for elem in resultList]) / len(testList)
        hit3 = sum([elem[1] for elem in resultList]) / len(testList)
        hit10 = sum([elem[2] for elem in resultList]) / len(testList)
        meanrank = sum([elem[3] for elem in resultList]) / len(testList)
        meanrerank = sum([elem[4] for elem in resultList]) / len(testList)
    else:
        hit1 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
        hit3 = sum([elem[1] for elem in resultList]) / (2 * len(testList))
        hit10 = sum([elem[2] for elem in resultList]) / (2 * len(testList))
        meanrank = sum([elem[3] for elem in resultList]) / (2 * len(testList))
        meanrerank = sum([elem[4] for elem in resultList]) / (2 * len(testList))

    print('Meanrank: %.6f' % meanrank)
    print('Meanrerank: %.6f' % meanrerank)
    print('Hit@1: %.6f' % hit1)
    print('Hit@3: %.6f' % hit3)
    print('Hit@10: %.6f' % hit10)

    return hit1, hit3, hit10, meanrank, meanrerank


def evaluation_batch(testList, tripleDict, model, ent_embeddings, L1_flag, filter, k=0, head=0):
    # embeddings are numpy like

    if k > len(testList):
        testList = random.choices(testList, k=k)
    elif k > 0:
        testList = random.sample(testList, k=k)

    L = []
    process_data(testList, tripleDict, model, ent_embeddings, L1_flag, filter, L, head)

    resultList = list(L)

    hit1 = sum([elem[0] for elem in resultList])
    hit3 = sum([elem[1] for elem in resultList])
    hit10 = sum([elem[2] for elem in resultList])
    meanrank = sum([elem[3] for elem in resultList])
    meanrerank = sum([elem[4] for elem in resultList])

    if head == 1 or head == 2:
        return hit1, hit3, hit10, meanrank, meanrerank, len(testList)
    else:
        return hit1, hit3, hit10, meanrank, meanrerank, 2 * len(testList)

