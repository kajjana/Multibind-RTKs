#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train', type=str)
parser.add_argument('test', type=str)
parser.add_argument('save_model', type=str)
args = parser.parse_args()

train = args.train
test = args.test
save = args.save_model

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
import random
import pickle
import timeit

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

import math
import time
import timeit
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


class Engine:
    def __init__(self, model, optimizer, device):
        self._model = model
        self._device = device
        self._optimizer = optimizer
    
    """Simple utility to see what is NaN"""
    def isNan(self, x):
        return x!=x
    
    def multiloss(self, output_vec, target_vec):
        #output_vec = [output[:,x] for x in range(len(output[0]))]
        #target_vec = [target[:,x] for x in range(len(target[0]))]
        criterion = torch.nn.MSELoss()
        criterionind = torch.nn.MSELoss(reduction = 'sum')
        mse_part = 0
        masks = dict()
        loss1 = dict()
        #print("target", target_vec)
        for x in range(0,len(target_vec)):
            masks[x] = self.isNan(target_vec[x])
            #masks[x] = tmpmasks>0
            if target_vec[x][~masks[x]].nelement() == 0:
                loss1[x] = [torch.sqrt(torch.tensor(1e-20)),torch.tensor(0.0)]
                continue
            else: # non nans
                mse_part += criterion(output_vec[x].squeeze()[~masks[x]],target_vec[x][~masks[x]])
                loss1[x] = [criterionind(output_vec[x].squeeze()[~masks[x]],target_vec[x][~masks[x]]), torch.sum(~masks[x], dtype=float)]
        loss_mean = mse_part
        loss_list = [loss_mean]
        for x in range(0, len(target_vec)):
            loss_list.append(loss1[x])
        return loss_list
    
    def loss_fn(self, outputs, targets):
        z_interaction = outputs
        t_interaction = targets
        t_interactioncuda = []
        for i in range(len(t_interaction[0])):
            t_interactioncuda.append(t_interaction[:,i].to(self._device).float().squeeze())
        loss_mean_list = self.multiloss(z_interaction, t_interactioncuda)
        return loss_mean_list
    
    def train(self, train_loader):
        self._model.train()
        loss_total = [0,[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        
        for data in train_loader:
            self._optimizer.zero_grad()
            inputs = data
            targets = data.y.to(self._device)
            outputs = self._model(inputs)
            allloss = self.loss_fn(outputs, targets)
            allloss[0].backward()
            self._optimizer.step()
            loss_total[0] += allloss[0].to('cpu').data.numpy()
            for i in range(len(allloss)-1):
                for j in range(2):
                    loss_total[i+1][j] += allloss[i+1][j].to('cpu').data.numpy()
        loss_mean = []
        loss_mean.append(loss_total[0]/len(train_loader))
        for i in range(len(loss_total)-1):
            loss_mean.append(loss_total[i+1][0]/loss_total[i+1][1])
        return loss_mean
    
    def evaluate(self, test_loader):
        self._model.eval()
        loss_total = [0,[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        
        for data in test_loader:
            inputs = data
            targets = data.y.to(self._device)
            outputs = self._model(inputs)
            allloss = self.loss_fn(outputs, targets)
            loss_total[0] += allloss[0].to('cpu').data.numpy()
            for i in range(len(allloss)-1):
                for j in range(2):
                    loss_total[i+1][j] += allloss[i+1][j].to('cpu').data.numpy()
        loss_mean = []
        loss_mean.append(loss_total[0]/len(test_loader))
        for i in range(len(loss_total)-1):
            loss_mean.append(loss_total[i+1][0]/loss_total[i+1][1])
        return loss_mean


# In[3]:


class Net(torch.nn.Module):
    def __init__(self, npnalayers, graphin, edgedim, predictnode, nfinglayers, fingdim, fingernode, fingerout, dropout, mlpout, mlpnode, device, deg):
        super(Net, self).__init__()
        
        self.device = device
        self.deg = deg
        self.npnalayers = npnalayers
        self.pnain = graphin
        self.edgedim = edgedim
        self.nfinglayers = nfinglayers
        self.fingdim = fingdim
        self.fingernode = fingernode
        self.fingerout = fingerout
        self.mlpout = mlpout
        self.mlpnode = mlpnode
        self.dropout = dropout
        self.predictnode = predictnode

        self.pnaModel1, self.pnaModel2, self.pnaModel3 = self.PNAlayers()
        self.model2 = self.mlp()
        self.model3 = self.fingerprint_deep()
        self.connected = self.fullyconnected()
        
        self.predict_property1 = nn.Linear(self.predictnode, 1)
        self.predict_property2 = nn.Linear(self.predictnode, 1)
        self.predict_property3 = nn.Linear(self.predictnode, 1)
        self.predict_property4 = nn.Linear(self.predictnode, 1)
        self.predict_property5 = nn.Linear(self.predictnode, 1)
        self.predict_property6 = nn.Linear(self.predictnode, 1)
        self.predict_property7 = nn.Linear(self.predictnode, 1)

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
        layers.append(Linear(self.mlpnode, self.mlpnode))
        layers.append(ReLU())
        layers.append(Linear(self.mlpnode, self.mlpout))
        mlp_net = Sequential(*layers)
        return mlp_net
        
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

    def fullyconnected(self):
        layers = []
        layers.append(Dropout(self.dropout))
        layers.append(Linear((self.mlpout)+(self.fingerout), self.predictnode))
        layers.append(ReLU())
        connected_net = Sequential(*layers)
        return connected_net

    def forward(self, inputs):
        x = inputs.x.type(torch.float).to(self.device)
        edge_index = inputs.edge_index.to(self.device)
        edge_attr = inputs.edge_attr.type(torch.float).to(self.device)
        batch = inputs.batch.to(self.device)
        fingerprint = inputs.fing.to(self.device)
        
        x = x.squeeze()
        for conv, batch_norm, relu in zip(self.pnaModel1, self.pnaModel2, self.pnaModel3):
            x = relu(batch_norm(conv(x, edge_index, edge_attr)))
        x = global_add_pool(x, batch)
        x = self.model2(x)
        finger = self.model3(fingerprint)
        dmr = torch.cat([x, finger], dim =1)
        dmr2 = self.connected(dmr)
        property1 = self.predict_property1(dmr2)
        property2 = self.predict_property2(dmr2)
        property3 = self.predict_property3(dmr2)
        property4 = self.predict_property4(dmr2)
        property5 = self.predict_property5(dmr2)
        property6 = self.predict_property6(dmr2)
        property7 = self.predict_property7(dmr2)
        return [property1, property2, property3, property4, property5, property6, property7]
    
    def __call__(self, inputs):
        return self.forward(inputs)


# In[4]:


#Model Training


# In[6]:


def model_training(params, save_model=False):
    cuda = True
    torch.manual_seed(42)
    random.seed(42)
    if cuda:
        torch.cuda.manual_seed(42)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 1000
    start = timeit.default_timer()

    with open ('./X/'+train, 'rb') as train_fp:
        Train_X = pickle.load(train_fp)
    with open ('./X/'+test, 'rb') as intest_fp:
        Intest_X = pickle.load(intest_fp)
    print('Training set: '+str(len(Train_X)))
    print('Internal-test set: '+str(len(Intest_X)))
    
    deg = torch.zeros(5, dtype=torch.long)
    for data in Train_X:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    batch = 16
    train_loader = DataLoader(Train_X, batch_size=batch, shuffle=True, drop_last=True)
    intest_loader = DataLoader(Intest_X, batch_size=batch, shuffle=True)
    print('Data has been loaded')
    
    Model = Net(npnalayers= params['npnalayers'],
                graphin = Train_X[0].x.shape[1],
                edgedim = Train_X[0].edge_attr.shape[1],
                predictnode= params['predictnode'], 
                nfinglayers= params['nfinglayers'], 
                fingdim=Train_X[0].fing.shape[1], 
                fingernode= params['fingernode'], 
                fingerout= params['fingerout'],
                dropout= params['dropout'], 
                mlpout= params['mlpout'], 
                mlpnode= params['mlpnode'],
                device=DEVICE,
                deg=deg
                 ).to(DEVICE)
    Optimizer = optim.Adam(Model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    Scheduler = ReduceLROnPlateau(Optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)
    eng = Engine(model=Model, optimizer=Optimizer, device=DEVICE)
    
    best_loss = np.inf
    best_valid = [np.inf]*8
    early_stopping_iter = 10
    num_mean_vals = 10
    early_stopping_counter = 0
    decay_interval = 10
    lr_decay = 0.5
    
    target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
    Results = {'epoch':[],
               'loss_total_train':[], 'train_erbB4_MSE':[], 'train_egfr_MSE':[], 'train_met_MSE':[],
               'train_alk_MSE':[], 'train_erbB2_MSE':[], 'train_ret_MSE':[], 'train_ros1_MSE':[],
               'loss_total_internal-test':[],'internal-test_erbB4_MSE':[], 'internal-test_egfr_MSE':[], 'internal-test_met_MSE':[],
               'internal-test_alk_MSE':[], 'internal-test_erbB2_MSE':[], 'internal-test_ret_MSE':[], 'internal-test_ros1_MSE':[]
              }
    keys_list = list(Results)
    
    for epoch in range(EPOCHS):
        if epoch  % decay_interval == 0:
            Optimizer.param_groups[0]['lr'] *= lr_decay
        
        train_loss = eng.train(train_loader)
        intest_loss = eng.evaluate(intest_loader)
        Scheduler.step(intest_loss[0]) #total loss intest
        
        typeformat = []
        typeformat.append(epoch)
        typeformat.extend(train_loss)
        typeformat.extend(intest_loss)
        print('epoch:%d\ntrain total loss: %.3f\ntask1: %.3f, task2:%.3f, task3:%.3f ,task4: %.3f, task5:%.3f, task6:%.3f, task7:%.3f\ninternal-test total loss: %.3f\nloss1: %.3f, loss2: %.3f, loss3: %.3f , loss4: %.3f, loss5: %.3f, loss6: %.3f, loss7: %.3f'%tuple(typeformat))

        
        for i in range(len(typeformat)):
            col = keys_list[i]
            if i == 0:
                Results[col].append(typeformat[i])
            else:
                Results[col].append(float('%.4f'%typeformat[i]))

        if epoch >= num_mean_vals:
            val_list = []
            for val in Results['loss_total_internal-test'][-num_mean_vals:]: #total loss intest
                val_list.append(val)
            avg = float('%.4f'%(np.mean(val_list)))
            if avg < best_loss and intest_loss[0] < best_loss:
                best_loss = avg
                early_stopping_counter = 0
                if save_model:
                    Result_df = pd.DataFrame(Results)
                    Result_df.to_csv('./Results/'+save.split('.')[0]+'_MSE_result.csv')
                    with open('./Model/deg_'+save.split('.')[0]+'.pkl', 'wb') as dg:
                        pickle.dump(deg, dg)
                    torch.save(Model.state_dict(), './Model/'+save)
            else:
                early_stopping_counter += 1
        
        if early_stopping_counter == early_stopping_iter:
            print('Early stopping')
            break
    
    end = timeit.default_timer()
    time = end - start
    print('Model training has finished in '+ str(float('%.4f'%(time/60)))+' mins')


# In[7]:


best_params = {'npnalayers': 4, 'predictnode': 723, 'nfinglayers': 4, 'fingernode': 658, 'dropout': 0.1298463545417025, 'fingerout': 677, 'mlpout': 989, 'mlpnode': 19, 'lr': 0.0006482020429373524, 'weight_decay': 1.5712715904764654e-05}
result = model_training(best_params, save_model=True)

