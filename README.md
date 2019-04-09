# Recurrent Event Network for Reasoning over Temporal Knowledge Graphs
This repository is the official PyTorch implementation of "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs".

Paper: [Recurrent Event Network for Reasoning over Temporal Knowledge Graph](), [Representation Learning on Graphs and Manifolds 2019](https://rlgm.github.io).

## Installation
Install PyTorch (>= 0.4.0) following the instuctions on the [official website](https://pytorch.org/)
Our code is based on Python3. 

## Train and Test
Before running, you should preprocess datasets.
```bash
python3 data/DATA_NAME/get_history.py
```

Then, we are ready to train and test.
```bash
python3 link_predict.py -d DATA_NAME --gpu 0 --model 0 --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epochs 20 --batch-size 1024
```

The default hyperparameters give the best performances.

**Model variants**
The user must specify a --model, the variants of which are described in detail in the paper:
- Attentive aggregator: --model 0
- Mean aggregator: --model 1
- GCN aggregator: --model 2


## Baselines
We use the following public codes for baselines
- TransE, DistMult: [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch)
- ComplEx: [Link](https://github.com/thunlp/OpenKE)
- RGCN: [Link](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn)
- ConvE: [Link](https://github.com/TimDettmers/ConvE)
- Know-Evolve: [Link](https://github.com/rstriv/Know-Evolve)
- HyTE: [Link](https://github.com/malllabiisc/HyTE)

We implemented TA-TransE, TA-DistMult, and TTransE. You can find [here](https://github.com/changlinzhang/dynamic-KG-basic/tree/lastest-combined)