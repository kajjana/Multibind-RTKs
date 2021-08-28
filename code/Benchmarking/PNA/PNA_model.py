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
from torch_geometric.nn import BatchNorm, global_add_pool, PNAConv
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
    def __init__(self, npnalayers, graphin, edgedim, predictnode, dropout, mlpnode, device, deg):
        super(Net, self).__init__()
        
        self.device = device
        self.deg = deg
        self.npnalayers = npnalayers
        self.pnain = graphin
        self.edgedim = edgedim
        self.mlpnode = mlpnode
        self.dropout = dropout
        self.predictnode = predictnode
        
        self.pnaModel1, self.pnaModel2, self.pnaModel3 = self.PNAlayers()
        self.predict_property1 = self.mlp()

    def PNAlayers(self):
        pna_layers = []
        pna_batch_norms = []
        pna_relus = []
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        for _ in range(self.npnalayers):
            conv = PNAConv(in_channels=self.pnain, out_channels=self.pnain,
                           aggregators=aggregators, scalers=scalers, deg=self.deg,
                           pre_layers=1, post_layers=1, edge_dim=self.edgedim, towers=5,
                           divide_input=False)
            pna_layers.append(conv)
            pna_batch_norms.append(BatchNorm(self.pnain))
            pna_relus.append(ReLU())
        pna_net = Sequential(*pna_layers)
        pna_batchnorm = Sequential(*pna_batch_norms)
        pna_relu = Sequential(*pna_relus)
        return pna_net, pna_batchnorm, pna_relu
        
    def mlp(self):
        layers = []
        layers.append(Linear(self.pnain, self.mlpnode))
        layers.append(ReLU())
        layers.append(Linear(self.mlpnode, self.predictnode))
        layers.append(ReLU())
        layers.append(Linear(self.predictnode, 1))
        mlp_net = Sequential(*layers)
        return mlp_net

    def forward(self, inputs):
        x = inputs.x.type(torch.float).to(self.device)
        edge_index = inputs.edge_index.to(self.device)
        edge_attr = inputs.edge_attr.type(torch.float).to(self.device)
        batch = inputs.batch.to(self.device)
        
        x = x.squeeze()
        for conv, batch_norm, relu in zip(self.pnaModel1, self.pnaModel2, self.pnaModel3):
            x = relu(batch_norm(conv(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch)
        property1 = self.predict_property1(x)
        return property1
    
    def __call__(self, inputs):
        return self.forward(inputs)

