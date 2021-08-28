#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def CreateDataset(fold,target):
    save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
    target = str(target)


    train_csv = save_dir+target+'/dataframe/cv/train_'+'kfold-'+str(fold)+'.csv'
    train = pd.read_csv(train_csv,index_col=0).reset_index(drop=True)
    train_pca_csv = save_dir+target+'/PCA/train_PCA16FPs.csv'
    train_pca = pd.read_csv(train_pca_csv,index_col=0).reset_index(drop=True)
    train = train[['SMILES_NS','pIC50']]
    train = pd.concat([train,train_pca],axis=1,join="inner").dropna().reset_index(drop=True)
    train_x = train.iloc[:, 3:].values  
    train_y = train.iloc[:, 1].values  

    valid_csv = save_dir+target+'/dataframe/cv/valid_'+'kfold-'+str(fold)+'.csv'
    valid = pd.read_csv(valid_csv,index_col=0).reset_index(drop=True)
    valid_pca_csv = save_dir+target+'/PCA/train_PCA16FPs.csv'
    valid_pca = pd.read_csv(valid_pca_csv,index_col=0).reset_index(drop=True)
    valid = valid[['SMILES_NS','pIC50']]
    valid = pd.concat([valid,valid_pca],axis=1,join="inner").dropna().reset_index(drop=True)
    valid_x = valid.iloc[:, 3:].values  
    valid_y = valid.iloc[:, 1].values  


    intest_csv = save_dir+target+'/dataframe/internal-test.csv'
    intest = pd.read_csv(intest_csv,index_col=0).reset_index(drop=True)
    intest_pca_csv = save_dir+target+'/PCA/internal-test_PCA16FPs.csv'
    intest_pca = pd.read_csv(intest_pca_csv,index_col=0).reset_index(drop=True)
    intest = intest[['SMILES_NS','pIC50']]
    intest = pd.concat([intest,intest_pca],axis=1,join="inner").dropna().reset_index(drop=True)
    intest_x = intest.iloc[:, 3:].values  
    intest_y = intest.iloc[:, 1].values  
    
    return train_x, train_y, valid_x, valid_y, intest_x, intest_y


# In[ ]:


def RF_model(fold, target, param):

    train_x, train_y, valid_x, valid_y, intest_x, intest_y = CreateDataset(fold=fold,target=target)
    
    rf = RandomForestRegressor(bootstrap=param['bootstrap'], ccp_alpha=0.0, criterion='mse',
                      max_depth=param['max_depth'], max_features=param['max_features'], max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=param['min_samples_leaf'],
                      min_samples_split=param['min_samples_split'], min_weight_fraction_leaf=0.0,
                      n_estimators=param['n_estimators'], n_jobs=1, oob_score=False,
                      verbose=0, warm_start=False)
    rf.fit(train_x,train_y)
    train_prediction = rf.predict(train_x)
    valid_prediction = rf.predict(valid_x)
    intest_prediction = rf.predict(intest_x)
    

    yhat = [item for item in valid_prediction]
    ytrue = [item for item in valid_y]
    result = {'ytrue':ytrue,'yhat':yhat}
    result_df = pd.DataFrame(result)
    save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
    result_df.to_csv(save_dir+target+'/dataframe/cv/Result/adjust_RFresult_valid_kfold-'+str(fold)+'.csv', index=False)  

    R2_train=r2_score(train_y, train_prediction)
    MAE_train=mean_absolute_error(train_y, train_prediction)
    MSE_train=mean_squared_error(train_y, train_prediction)

    R2_valid=r2_score(valid_y, valid_prediction)
    MAE_valid=mean_absolute_error(valid_y, valid_prediction)
    MSE_valid=mean_squared_error(valid_y, valid_prediction)

    R2_intest=r2_score(intest_y, intest_prediction)
    MAE_intest=mean_absolute_error(intest_y, intest_prediction)
    MSE_intest=mean_squared_error(intest_y, intest_prediction)

    R2_train_list.append(R2_train)
    MAE_train_list.append(MAE_train)
    MSE_train_list.append(MSE_train)

    Q2_valid_list.append(R2_valid)
    MAE_valid_list.append(MAE_valid)
    MSE_valid_list.append(MSE_valid)

    Q2_intest_list.append(R2_intest)
    MAE_intest_list.append(MAE_intest)
    MSE_intest_list.append(MSE_intest)
        

    return R2_train, MAE_train, MSE_train, R2_valid, MAE_valid, MSE_valid, R2_intest, MAE_intest, MSE_intest, train_x, valid_x, intest_x


# In[ ]:


def print_output(fold, R2_train, MAE_train, MSE_train, R2_valid, MAE_valid, MSE_valid, R2_intest, MAE_intest, MSE_intest, train_x, valid_x, intest_x):
    
    print('\nFold: ',fold)

    print('\nTraining set\n------------')
    print('N: '  +str(len(train_x)))
    print('R2: %0.4f'%(R2_train))
    print('MAE: %0.4f'%(MAE_train))
    print('MSE: %0.4f'%(MSE_train))

    print('\nValidation set\n------------')
    print('N: ' +str(len(valid_x)))
    print('R2: %0.4f'%(R2_valid))
    print('MAE: %0.4f'%(MAE_valid))
    print('MSE: %0.4f'%(MSE_valid))

    print('\nIntest set\n------------')
    print('N: ' +str(len(intest_x)))
    print('R2: %0.4f'%(R2_intest))
    print('MAE: %0.4f'%(MAE_intest))
    print('MSE: %0.4f'%(MSE_intest))


# In[ ]:


import time
import timeit
start = timeit.default_timer()

target_list = ['erbB2','ret','met','ros1']
param_list = [alk_param,egfr_param,erbB4_param,erbB2_param,ret_param,met_param,ros1_param]
for tar,param in zip(target_list,param_list):
    print('TARGET:',tar)
    for fold in range(10):
        R2_train, MAE_train, MSE_train, R2_valid, MAE_valid, MSE_valid, R2_intest, MAE_intest, MSE_intest, train_x, valid_x, intest_x = RF_model(fold, target=tar,param=param)
        print_output(fold, R2_train, MAE_train, MSE_train, R2_valid, MAE_valid, MSE_valid, R2_intest, MAE_intest, MSE_intest, train_x, valid_x, intest_x)   
end = timeit.default_timer()
time = end - start
print("Time used:",time)

