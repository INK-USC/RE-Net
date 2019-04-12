import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

class marginLoss(nn.Module):
	def __init__(self):
		super(marginLoss, self).__init__()

	def forward(self, pos, neg, margin):
		zero_tensor = floatTensor(pos.size())
		zero_tensor.zero_()
		zero_tensor = autograd.Variable(zero_tensor)
		return torch.sum(torch.max(pos - neg + margin, zero_tensor))

def orthogonalLoss(rel_embeddings, norm_embeddings):
	return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):
	norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
	return torch.sum(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))

def regulLoss(embeddings):
	return torch.mean(embeddings ** 2)

class binaryCrossLoss(nn.Module):
	def __init__(self):
		super(binaryCrossLoss, self).__init__()

	def forward(self, pos, neg):
		pos_labels = floatTensor(pos.shape[0])
		nn.init.ones_(pos_labels)
		neg_labels = floatTensor(neg.shape[0])
		nn.init.zeros_(neg_labels)
		labels = torch.cat((pos_labels, neg_labels))
		return F.binary_cross_entropy_with_logits(torch.cat((pos, neg)), labels)
