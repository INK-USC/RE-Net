# Recurrent Event Network for Reasoning over Temporal Knowledge Graphs
This repository is the official PyTorch implementation of "Recurrent Event Network for Reasoning over Temporal Knowledge Graphs".

Paper: [Recurrent Event Network for Reasoning over Temporal Knowledge Graph](), [Representation Learning on Graphs and Manifolds 2019](https://rlgm.github.io).

## Installation
Install PyTorch (>= 0.4.0) following the instuctions on the [official website](https://pytorch.org/)

## Train and Test
Before running, you should preprocess datasets.
```bash
python data/DATA_NAME/get_history.py
```

Then, we are ready to train and test
```bash
python link_predict.py -d DATA_NAME --gpu 0 --model 0 
```

Each model has different aggregators.
Model 0 uses the attentive aggregator, model 1 uses the mean aggregator, and model 2 uses the GCN aggregator.
