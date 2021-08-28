#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import skopt
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[ ]:


import pandas as pd
import numpy as np

def CreateDataset(target):
    save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
    target = str(target)


    train_csv = save_dir+target+'/train.csv'
    train = pd.read_csv(train_csv,index_col=0).reset_index(drop=True)
    train_pca_csv = save_dir+target+'/PCA/train_PCA16FPs.csv'
    train_pca = pd.read_csv(train_pca_csv,index_col=0).reset_index(drop=True)
    train = train[['SMILES_NS','pIC50']]
    train = pd.concat([train,train_pca],axis=1,join="inner").dropna().reset_index(drop=True)
    train_x = train.iloc[:, 3:].values  
    train_y = train.iloc[:, 1].values  

    intest_csv = save_dir+target+'/internal-test.csv'
    intest = pd.read_csv(intest_csv,index_col=0).reset_index(drop=True)
    intest_pca_csv = save_dir+target+'/PCA/internal-test_PCA16FPs.csv'
    intest_pca = pd.read_csv(intest_pca_csv,index_col=0).reset_index(drop=True)
    intest = intest[['SMILES_NS','pIC50']]
    intest = pd.concat([intest,intest_pca],axis=1,join="inner").dropna().reset_index(drop=True)
    intest_x = intest.iloc[:, 3:].values  
    intest_y = intest.iloc[:, 1].values  

    return train_x, train_y, intest_x, intest_y 


# In[ ]:


from skopt.space import Real, Categorical, Integer
target_list = ['ros1']
BS = []
BP = []
i=0

def on_step(optim_result):
    global i
    i+=1
    mse = search.best_score_
    print('best_score: ',mse)
    print(i,search.best_params_)

for tar in target_list:
    train_x, train_y, intest_x, intest_y= CreateDataset(target=tar)
    print(len(train_x),len(intest_x))
    params = {
    'bootstrap': Categorical([True, False]),
    'max_depth': Integer(2, 10),
    'max_features': Categorical(['sqrt','auto']),
    'min_samples_leaf': Integer(2, 5),
    'min_samples_split': Integer(2, 5),
    'n_estimators': Integer(100, 1000)
    }
    # define the search
    search = BayesSearchCV(estimator=RandomForestRegressor(), search_spaces=params,scoring = 'neg_mean_squared_error', n_jobs=1,n_iter=50, cv=10,)
    # perform the search
    search.fit(train_x, train_y,callback=on_step)
    # report the best result
    print(search.best_score_)
    print(search.best_params_)
    BS.append(search.best_score_)
    BP.append(search.best_params_)
    
Tune = {'target':target_list,'best_score':BS,'best_params':BP}
result_df = pd.DataFrame(Tune)
result_df.to_csv('/share/galaxy/fahsai/NSCLC/Sin/RF_tuneparam.csv', index=False)

