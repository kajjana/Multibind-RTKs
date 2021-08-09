#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8


import argparse
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.nn import Sequential, ReLU, Linear, Dropout
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool
from torch_geometric.data import DataLoader

from sklearn.metrics import mean_squared_error

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


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


#similarity-based functions

def getECFP(smiles_list, rad):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    fp = [AllChem.GetMorganFingerprint(mol,rad) for mol in mols]
    return fp

def getTanimotoSim(train_fp, test_fp):
    index_name = []
    for ind in range(len(test_fp)):
        index = 'valid_'+str(ind)
        index_name.append(index)
    col_name = []
    for c in range(len(train_fp)):
        col = 'train_'+str(c)
        col_name.append(col)
    test_sim_df = pd.DataFrame(index=index_name, columns=col_name)
    for i in range(len(test_fp)):
        if (i%10 ==0 or i == len(test_fp)-1): 
            print('Compound-',i)
        ref_fp = test_fp[i]
        for j in range(len(train_fp)):
            com_fp = train_fp[j]
            sim = DataStructs.TanimotoSimilarity(ref_fp, com_fp)
            test_sim_df.iloc[i,j] = float('%.4f'%sim)
    print("Similarity Measurement Finish \n")        
    return test_sim_df


parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('predict_dataset', type=str)
parser.add_argument("AD", type=str,help="Include AD analysis")
args = parser.parse_args()

model = args.model
dataset = args.predict_dataset
AD=args.AD



with open ('./X/'+dataset, 'rb') as dataset_fp:
    dataset_X = pickle.load(dataset_fp)
print('Predict dataset: '+str(len(dataset_X)))

TK_prediction = predictTK(Net, dataset_X)

dataset_df = pd.read_csv(dataset.split('.')[0]+'.csv', index_col = 0)
for tar in TK_prediction:
    dataset_df['predicted_pIC50_'+tar] = TK_prediction.get(tar)
dataset_result = dataset_df

print("......Export Predicted Results......")
save_csv = './Results/Result_'+dataset.split('.')[0]+'.csv'
dataset_result.to_csv(save_csv)

if AD == 'AD':
    print("Start AD analysis")
    target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
    train_df = pd.read_csv('./AD/train_for_AD.csv', index_col = 0)
    test_pred = pd.read_csv(dataset[:-4]+'.csv',index_col=0)

    #fingerprint-based similarity measurement
    print('fingerprint-based similarity measurement')
    train_fp = getECFP(train_df.smiles, 4)
    test_fp = getECFP(test_pred.smiles, 4)
    test_sim = getTanimotoSim(train_fp, test_fp)
    
    #Caculation of AD parameters
    cv_tk = pd.read_csv('./AD/CVprediction_for_AD.csv',index_col=0)
    test_sim = test_sim
    sim_list = []
    nb_RMSE_list = []
    for ind in range(len(test_sim.index)):
        sim = test_sim.iloc[ind]
        max_sim = max(sim)
        target_RMSE_nb = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        sim_list.append(max_sim)
        if len(list(sim.index[sim >= 0.35].values)) != 0:
            nb = list(sim.index[sim >= 0.35].values)
            n_nb = len(nb)
            nb_index = [int(idx[6:]) for idx in nb]
            nb_smiles = list(train_df.iloc[nb_index,0])
            nb_cv = cv_tk[cv_tk.smiles.isin(nb_smiles)]
            for t,tar in enumerate(target_list):
                tar_nb_cv = nb_cv[nb_cv.target == tar]
                if tar_nb_cv.shape[0] != 0:
                    target_RMSE_nb[t] = round(mean_squared_error(tar_nb_cv.pIC50, tar_nb_cv.predicted_pIC50, squared=False),4)
            nb_RMSE_list.append(target_RMSE_nb)

        else:
            nb_RMSE_list.append([np.nan]*7)

    test = pd.read_csv(save_csv,index_col=0)        
    test['similarity'] = sim_list
    test['RMSE_neighbors'] = nb_RMSE_list
    test[['RMSE_erbB4_neighbors','RMSE_egfr_neighbors','RMSE_met_neighbors','RMSE_alk_neighbors',
          'RMSE_erbB2_neighbors','RMSE_ret_neighbors','RMSE_ros1_neighbors']] = pd.DataFrame(test.RMSE_neighbors.tolist(), index=test.index)
    test.drop(columns=['RMSE_neighbors'],inplace=True)
    
    AD_param = {'sim': 0.54886, 'error': {'erbB4': 0.54, 'egfr': 0.84, 'met': 0.654, 'alk': 0.594, 
                                          'erbB2': 0.6, 'ret': 0.64, 'ros1': 0.513}}
    inAD_smi = {}
    for t,tar in enumerate(target_list):
        inAD = test[(test.similarity >= AD_param['sim'])&(test['RMSE_'+tar+'_neighbors'] < AD_param['error'][tar])]
        print('inside',tar,'domain:',inAD.shape[0])
        inAD_smi[tar] = inAD.smiles
        inAD = inAD.copy()
        try:
            inAD.loc[:,tar+'_domain'] = 'inside'
        except:
            inAD[tar+'_domain'] = 'outside'
        if t == 0:
            tk_inAD = inAD
        else:
            tk_inAD = pd.merge(tk_inAD,inAD, how='outer')

    test_pred = pd.read_csv(save_csv,index_col=0)
    test_AD = pd.merge(test_pred,tk_inAD, how='outer')
    for tar in target_list:
        test_AD.replace({tar+'_domain': {np.nan: 'outside'}},inplace=True)
    col_name = ['RMSE_'+tar+'_neighbors' for tar in target_list]
    test_AD.drop(columns=col_name, inplace=True)
    test_AD.drop(columns=['similarity'], inplace=True)
    test_AD.to_csv(save_csv)
    
    print("\nFinish AD analysis")
else:
    print('No AD analysis')
    print('Finish')




