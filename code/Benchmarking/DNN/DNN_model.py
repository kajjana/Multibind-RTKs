#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, Dropout
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, global_add_pool
from torch_geometric.data import Data
import torch.optim as optim

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

import rdkit
from rdkit import Chem

import numpy as np
import pandas as pd
import math


# In[ ]:


class Engine:
    def __init__(self, model, optimizer, device):
        self._model = model
        self._device = device
        self._optimizer = optimizer
        self._criterion = nn.MSELoss(reduction='sum')
    
    def train(self, train_loader):
        self._model.train()
        loss_total = 0
        
        for data in train_loader:
            self._optimizer.zero_grad()
            inputs = data
            targets = data.y.to(self._device)
            outputs = self._model(inputs)
            loss = self._criterion(outputs.float(), targets.float())
            loss.backward()
            self._optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        loss_mean = loss_total/len(train_loader.dataset)
        return loss_mean
    
    def evaluate(self, test_loader):
        self._model.eval()
        loss_total = 0
        for data in test_loader:
            inputs = data
            targets = data.y.to(self._device)
            outputs = self._model(inputs)
            loss = self._criterion(outputs.float(), targets.float())
            loss_total += loss.to('cpu').data.numpy()
        loss_mean = loss_total/len(test_loader.dataset)
        return loss_mean


# In[ ]:


class Net(torch.nn.Module):
    def __init__(self, nfinglayers, fingdim , fingernode, fingerout, dropout, device):
        super(Net, self).__init__()
        
        self.device = device
        self.nfinglayers = nfinglayers
        self.fingdim = fingdim
        self.fingernode = fingernode
        self.fingerout = fingerout
        self.dropout = dropout

        self.model1 = self.fingerprint_deep()
        self.predict_property1 = nn.Linear(self.fingerout, 1)

    def fingerprint_deep(self):
        fing_layers = []
        for _ in range(self.nfinglayers):
            if len(fing_layers) == 0:
                fing_layers.append(Linear(self.fingdim, self.fingernode))
                fing_layers.append(BatchNorm(self.fingernode))
                fing_layers.append(Dropout(self.dropout))
                fing_layers.append(ReLU())
            else:
                fing_layers.append(Linear(self.fingernode, self.fingernode))
                fing_layers.append(BatchNorm(self.fingernode))
                fing_layers.append(Dropout(self.dropout))
                fing_layers.append(ReLU())
        fing_layers.append(Linear(self.fingernode, self.fingerout))
        fing_net = Sequential(*fing_layers)
        return fing_net

    def forward(self, inputs):
        fingerprint = inputs.fing.to(self.device)
        finger = self.model1(fingerprint)
        property1 = self.predict_property1(finger)
        return property1
    
    def __call__(self, inputs):
        return self.forward(inputs)

