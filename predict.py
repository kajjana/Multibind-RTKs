#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('predict_dataset', type=str)
args = parser.parse_args()

model = args.model
dataset = args.predict_dataset

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear, Dropout
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
from torch_geometric.data import DataLoader

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


# In[2]:


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


# In[3]:


def predictTK(modelClass, dataset):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    f = open('./Model/deg_'+model.split('.')[0]+'.pkl', 'rb')
    deg = pickle.load(f)
    f.close()    

    params =  {'npnalayers': 4, 'predictnode': 723, 'nfinglayers': 4, 'fingernode': 658, 'dropout': 0.1298463545417025, 'fingerout': 677, 'mlpout': 989, 'mlpnode': 19, 'lr': 0.0006482020429373524, 'weight_decay': 1.5712715904764654e-05}
    
    Model = Net(npnalayers= params['npnalayers'],
                graphin = dataset[0].x.shape[1],
                edgedim = dataset[0].edge_attr.shape[1],
                predictnode= params['predictnode'], 
                nfinglayers= params['nfinglayers'], 
                fingdim= dataset[0].fing.shape[1], 
                fingernode= params['fingernode'], 
                fingerout= params['fingerout'],
                dropout= params['dropout'], 
                mlpout= params['mlpout'], 
                mlpnode= params['mlpnode'],
                device=DEVICE,
                deg=deg
                 ).to(DEVICE)
    Model.load_state_dict(torch.load('./Model/'+model))
    prediction = Model.eval()
    
    tk_predicted = {'erbB4':[], 'egfr':[], 'met':[], 'alk':[], 'erbB2':[], 'ret':[], 'ros1':[]}
    for i, data in enumerate(data_loader):
        predicted = prediction(data)
        for t, v in enumerate(predicted):
            val = v.cpu().detach().numpy().tolist()
            pred_val = float('%.2f'%val[0][0])
            tk_predicted[list(tk_predicted.keys())[t]].append(pred_val)
    return tk_predicted


# In[5]:


with open ('./X/'+dataset, 'rb') as dataset_fp:
    dataset_X = pickle.load(dataset_fp)
print('Predict dataset: '+str(len(dataset_X)))

TK_prediction = predictTK(Net, dataset_X)

dataset_df = pd.read_csv(dataset.split('.')[0]+'.csv', index_col = 0)
for tar in TK_prediction:
    dataset_df['predicted_pIC50_'+tar] = TK_prediction.get(tar)
dataset_result = dataset_df

print("......Export Predicted Results......")
dataset_result.to_csv('./Results/Result_'+dataset.split('.')[0]+'.csv')








