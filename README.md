## PyTorch implementation of Recurrent Event Network (RE-Net)

Paper: [Recurrent Event Network for Reasoning over Temporal Knowledge Graph]()

[ICLR Workshop on Representation Learning on Graphs and Manifolds](https://rlgm.github.io), 2019.

## Installation
Install PyTorch (>= 0.4.0) following the instuctions on the [official website](https://pytorch.org/).
Our code is written on Python3. 

## Train and Test
Before running, you should preprocess datasets.
```bash
python3 data/DATA_NAME/get_history.py
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
- GCN aggregator: --model 2

### Data
There are two datasets: ICEWS18, GDELT.
Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt', and 'get_history.py'.
- 'get_history': This is for getting history for each entity.
- 'stat.txt': First value is the number of entities, and second value is the number of relations.
- 'train.txt', 'valid.txt', 'test.txt': First column is subject entities, second column is relations, and third column is object entities. The fourth column is time.

### Predictive performances
In the ICEWS18 datasets, the results with filtered metrics:

| Method        | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------------|-------|--------|--------|---------|
| RE-Net (mean) | 42.38 | 35.80  | 44.99  | 54.90   |
| RE-Net (Attn) | 41.46 | 34.67  | 44.19  | 54.44   |
| RE-Net (GCN)  | 41.35 | 34.53  | 44.05  | 54.35   |

In the GDELT datasets, the results with filtered metrics:

| Method        | MRR   | Hits@1 | Hits@3 | Hits@10 |
|---------------|-------|--------|--------|---------|
| RE-Net (mean) | 39.15 | 30.84  | 43.07  | 53.48   |
| RE-Net (Attn) | 38.07 | 29.44  | 42.26  | 52.93   |
| RE-Net (GCN)  | 37.99 | 30.05  | 41.40  | 52.18   |

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


<!-- We use the following public codes for baselines
- TransE, DistMult: [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch)
- ComplEx: [Link](https://github.com/thunlp/OpenKE)
- RGCN: [Link](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn)
- ConvE: [Link](https://github.com/TimDettmers/ConvE)
- Know-Evolve: [Link](https://github.com/rstriv/Know-Evolve)
- HyTE: [Link](https://github.com/malllabiisc/HyTE)
 -->
We implemented TA-TransE, TA-DistMult, and TTransE. The user can find [here](https://github.com/changlinzhang/dynamic-KG-basic/tree/lastest-combined).

## Related Work
There are related literatures: Temporal Knowledge Graph Embedding, Dynamic Graph Embedding, Knowledge Graph Embedding, Static Graph Embedding, etc.
We organized the list of [related work](https://github.com/woojeongjin/dynamic-KG).
