import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import os

class MultiBinaryCrossentropy(nn.Module):
    def __init__(self, categories):
        super(MultiBinaryCrossentropy, self).__init__()
        self.categories = categories
        self.pred_splits = [1] * self.categories


    def forward(self, y_pred, y_true):
        loss_per_category = []
        
        labels_per_category = torch.unbind(y_true, axis=1)
        preds_per_category = torch.split(y_pred, self.pred_splits, dim=1)
        for labels, preds in zip(labels_per_category, preds_per_category):
            preds = preds.reshape((preds.shape[0],))
            loss_per_category.append(F.binary_cross_entropy(labels, preds))
        return torch.sum(torch.tensor(loss_per_category))
    