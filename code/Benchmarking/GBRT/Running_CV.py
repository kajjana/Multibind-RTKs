#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


# In[ ]:


target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
target = target_list[0] #erbB4
print(target, 'model running')


# In[ ]:


result = {}
data_dir = '/'+target
for fold in range(10):
    print('Fold-',fold)
    
    print('Train data')
    X_train_df =  pd.read_csv(data_dir+'/PCA/cv/train_kfold-'+str(fold)+'.csv',index_col=0)
    print(X_train_df.shape)
    x_train = X_train_df.iloc[:,1:]
    print(x_train.shape)
    
    Y_train_df = pd.read_csv(data_dir+'/dataframe/cv/train_kfold-'+str(fold)+'.csv', index_col=0)
    print(Y_train_df.shape)
    y_train = Y_train_df[['pIC50']]
    print(y_train.shape)
    
    print('Valid data')
    X_test_df = pd.read_csv(data_dir+'/PCA/cv/valid_kfold-'+str(fold)+'.csv', index_col=0)
    print(X_test_df.shape)
    x_test = X_test_df.iloc[:,1:]
    print(x_test.shape)

    Y_test_df = pd.read_csv(data_dir+'/dataframe/cv/valid_kfold-'+str(fold)+'.csv', index_col=0)
    print(Y_test_df.shape)
    y_test = Y_test_df[['pIC50']]
    print(y_test.shape)
    
    result[fold] = x_train,x_test,y_train,y_test
    
    #using the best parameter set from hyperparameter tuning
    #example erbB4 best parameters
    model = XGBRegressor(learning_rate=0.009931147575,max_depth = 8,max_features = 'sqrt',min_samples_leaf = 6,min_samples_split= 4,n_estimators = 871)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(y_pred)

