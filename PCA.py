#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train', type=str)
parser.add_argument('dataset_FP', type=str)
parser.add_argument('save', type=str)
args = parser.parse_args()

train = args.train
FP = args.dataset_FP
save = args.save

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt


# In[2]:


def fitPCA(fp_df, model):
    print(fp_df.shape)
    result = model.transform(fp_df)
    print(result.shape)
    return result

def concatResult(smi_df, pca_df):
    pca_columns = []
    for i in range(len(pca_df[0])):
        col = 'pca_'+str(i)
        pca_columns.append(col)
    principalDf = pd.DataFrame(data = pca_df, columns = pca_columns)
    concat_df = pd.concat([smi_df, principalDf], axis=1)
    return concat_df


# In[3]:


train = pd.read_csv(train,index_col=0)
fp = pd.read_csv(FP, index_col=0)
dataset =  pd.merge(train, fp, on=['smiles'], how='inner')


# In[4]:


#all train data
train_fp = dataset[dataset.columns[8:]]
print('Data Shape')
print('Before PCA',train_fp.shape)

# at %95 variance
n=0.95
pca_model = PCA(n_components=n)
train_pca = pca_model.fit_transform(train_fp)
print('After PCA',train_pca.shape)


# In[5]:


import pickle
#Save trained PCA model
print('......Save PCA Model......')
pkl_filename = "./Model/"+save
with open(pkl_filename, 'wb') as file:
    pickle.dump(pca_model, file)


# In[7]:


print('Finished')


# In[ ]:




