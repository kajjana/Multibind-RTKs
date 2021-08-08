#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('dataset_FP', type=str)
parser.add_argument('pca_model', type=str)
parser.add_argument('task', type=str)
args = parser.parse_args()

data = args.dataset
FP = args.dataset_FP
pca = args.pca_model
task = args.task

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

import torch
from torch_geometric.data import Data
import rdkit
from rdkit import Chem
import deepchem as dc


# In[4]:


def fitPCA(fp_df, model):
    #print(fp_df.shape)
    result = model.transform(fp_df)
    #print(result.shape)
    return result

def concatResult(smi_df, pca_df):
    pca_columns = []
    for i in range(len(pca_df[0])):
        col = 'pca_'+str(i)
        pca_columns.append(col)
    principalDf = pd.DataFrame(data = pca_df, columns = pca_columns)
    concat_df = pd.concat([smi_df, principalDf], axis=1)
    return concat_df


# In[5]:


import pickle
pkl_filename = "./Model/"+pca
with open(pkl_filename, 'rb') as file:
    trainedPCA = pickle.load(file)


# In[9]:


def PCA_gen(data,FP):
    fp = pd.read_csv(FP, index_col=0).reset_index(drop=True)
    dataset = pd.read_csv(data, index_col=0)
    dataset =  pd.merge(dataset, fp, on=['smiles'], how='inner')
    dataset_fp = dataset[dataset.columns[-30562:]]
    dataset_pca = fitPCA(dataset_fp, trainedPCA)
    dataset_pca_df = concatResult(dataset[['smiles']], dataset_pca)
    dataset_pca_df.to_csv('./PCA_FP/'+ data[:-4] +'_PCA16FPs.csv')
   

target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']

def torchData(data):
    X =  Data(x=torch.tensor(data.node_features, dtype=torch.float),
              edge_index=torch.tensor(data.edge_index, dtype=torch.long),
              edge_attr=torch.tensor(data.edge_features,dtype=torch.float)
             )
    return X

def loadData(df, index):
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    out = featurizer.featurize(df.smiles)
    X = [torchData(out[i]) for i in index]
    fingerprint = [df.iloc[i, -1460:].values.astype(np.float).tolist() for i in index]
    for ind, data in enumerate(X):
        fing = fingerprint[ind]
        data.fing = torch.tensor([fing])
        if task == 'Train':
            y1 = df.pIC50_erbB4[index[ind]]
            y2 = df.pIC50_egfr[index[ind]]
            y3 = df.pIC50_met[index[ind]]
            y4 = df.pIC50_alk[index[ind]]
            y5 = df.pIC50_erbB2[index[ind]]
            y6 = df.pIC50_ret[index[ind]]
            y7 = df.pIC50_ros1[index[ind]]
            data.y = torch.tensor([[y1,y2,y3,y4,y5,y6,y7]])
        elif task == 'Screen':
            pass
        else :
            import sys 
            print('Cannot do this task')
            sys.exit()
    return X 


# In[14]:

print('......PCA fitting......')
PCA_gen(data,FP)

print('......Feature Generating......')
pca_fp_file = './PCA_FP/'+ data[:-4] + '_PCA16FPs.csv'
pca_df = pd.read_csv(pca_fp_file, index_col=0)
data = pd.read_csv(data, index_col=0)
df = data.merge(pca_df, on=['smiles'])
data_X = loadData(df, df.index)

data = args.dataset
save_feature = './X/'+data[:-4]+'.pkl'
print(save_feature)
with open(save_feature, 'wb') as fp:
    pickle.dump(data_X, fp)


# In[15]:


print('Finish Loading')


# In[ ]:




