## PyTorch implementation of Recurrent Event Network (RE-Net)

Paper: [Recurrent Event Network for Reasoning over Temporal Knowledge Graph](https://arxiv.org/abs/1904.05530)

This repository contains the implementation of the RE-Net architectures described in the paper.

<img src="figs/renet.png" width="400" align="middle"/>

Recently, there has been a surge of interest in learning representation of graph-structured data that are dynamically evolving. However, current dynamic graph learning methods lack a principled way in modeling temporal, multi-relational, and concurrent interactions between nodes—a limitation that is especially problematic for the task of temporal knowledge graph reasoning, where the goal is to predict unseen entity relationships (i.e., events) over time. Here we present Recurrent Event Network (RE-Net)—a novel neural architecture for modeling complex event sequences—which consists of a recurrent event encoder and a neighborhood aggregator.

## Installation
Install PyTorch (>= 0.4.0) and DGL following the instuctions on the [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai).
Our code is written in Python3.

## Train and Test
Before running, you should preprocess datasets.

For attentive, mean, pooling aggregators (model 0,1,2)
```bash
python3 data/DATA_NAME/get_history.py
```

For an RGCN aggregator (model 3)
```bash
python3 data/DATA_NAME/get_history_graph.py
```

Then, we are ready to train and test.
We first train the model.
```bash
python3 train.py -d DATA_NAME --gpu 0 --model 0 --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epochs 20 --batch-size 1024
```

We are ready to test!
```bash
python3 test.py -d DATA_NAME --gpu 0 --model 0 --n-hidden 200
```

The default hyperparameters give the best performances.

### Model variants
The user must specify a --model, the variants of which are described in detail in the paper:
- Attentive aggregator: --model 0
- Mean aggregator: --model 1
- Pooling aggregator: --model 2
- RGCN aggregator: --model 3

## Related Work
There are related literatures: Temporal Knowledge Graph Embedding, Dynamic Graph Embedding, Knowledge Graph Embedding, Static Graph Embedding, etc.
We organized the list of [related work](https://github.com/woojeongjin/dynamic-KG).

## Datasets
There are four datasets: [ICEWS18](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/28075/Z1ZFYG&version=25.0), [GDELT](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/), [WIKI](https://www.wikidata.org/wiki/Wikidata:Main_Page), and [YAGO](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/).
Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt', 'get_history.py', and 'get_history_graph.py'.
- 'get_history.py': This is for getting history for model 0, 1, and 2.
- 'get_history_graph.py': This is for getting history and graph for model 3.
- 'stat.txt': First value is the number of entities, and second value is the number of relations.
- 'train.txt', 'valid.txt', 'test.txt': First column is subject entities, second column is relations, and third column is object entities. The fourth column is time.

## Baselines
We use the following public codes for baselines and hyperparameters. We validated embedding sizes among presented values.

| Baselines   | Code                                                                      | Embedding size | Batch size |
|-------------|---------------------------------------------------------------------------|----------------|------------|
| TransE      | [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch) | 100, 200       | 1024       |
| DistMult    | [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch) | 100, 200       | 1024       |
| ComplEx     | [Link](https://github.com/thunlp/OpenKE)                                  | 50, 100, 200   | 100        |
| RGCN        | [Link](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn)     | 200            | Default    |
| ConvE       | [Link](https://github.com/TimDettmers/ConvE)                              | 200            | 128        |
| Know-Evolve | [Link](https://github.com/rstriv/Know-Evolve)                             | Default        | Default    |
| HyTE        | [Link](https://github.com/malllabiisc/HyTE)                               | 128            | Default    |


We implemented TA-TransE, TA-DistMult, and TTransE. The user can find implementations in the 'baselines' folder.

## Predictive performances
In the **ICEWS18** dataset, the results with **filtered** metrics:

| Method        | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------------|-------|--------|--------|---------|
| RE-Net (mean) | 42.38 | 35.80  | 44.99  | 54.90   |
| RE-Net (attn) | 41.46 | 34.67  | 44.19  | 54.44   |
| RE-Net (pool) | 41.35 | 34.53  | 44.05  | 54.35   |
| RE-Net (RGCN) | 43.20 | 36.63  | 45.58  | 55.91   |

In the **GDELT** dataset, the results with **filtered** metrics:

| Method        | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------------|-------|--------|--------|---------|
| RE-Net (mean) | 39.15 | 30.84  | 43.07  | 53.48   |
| RE-Net (attn) | 38.07 | 29.44  | 42.26  | 52.93   |
| RE-Net (pool) | 37.99 | 30.05  | 41.40  | 52.18   |
| RE-Net (RGCN) | 40.21 | 32.54  | 43.53  | 53.83   |

In the **WIKI** dataset, the results with **filtered** metrics:

| Method        | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------------|-------|--------|--------|---------|
| RE-Net (mean) | 48.30 | 45.86  | 49.36  | 53.03   |
| RE-Net (attn) | 51.72 | 50.60  | 52.12  | 53.72   |
| RE-Net (pool) | 45.15 | 41.41  | 46.98  | 52.57   |
| RE-Net (RGCN) | 50.47 | 49.80  | 52.03  | 53.16   |

In the **YAGO** dataset, the results with **filtered** metrics:

| Method        | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------------|-------|--------|--------|---------|
| RE-Net (mean) | 65.51 | 63.85  | 66.06  | 68.03   |
| RE-Net (attn) | 65.79 | 64.50  | 66.00  | 67.82   |
| RE-Net (pool) | 63.65 | 61.25  | 64.76  | 67.45   |
| RE-Net (RGCN) | 65.69 | 64.83  | 66.32  | 68.48   |
