#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def predictionSum(df):
    target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
    summary = {'target':[], 'MAE':[], 'MSE':[], 'RMSE':[], 'R2':[]}
    tar = 'overall'
    summary['target'].append(tar)
    MAE = mean_absolute_error(df.pIC50, df.predicted_pIC50)
    summary['MAE'].append(MAE)
    MSE = mean_squared_error(df.pIC50, df.predicted_pIC50)
    summary['MSE'].append(MSE)
    root_MSE = mean_squared_error(df.pIC50, df.predicted_pIC50, squared=False)
    summary['RMSE'].append(root_MSE)
    R2 = r2_score(df.pIC50, df.predicted_pIC50)
    summary['R2'].append(R2)

    for tar in target_list:
        summary['target'].append(tar)
        tar_df = df[df.target == tar]
        MAE = mean_absolute_error(tar_df.pIC50, tar_df.predicted_pIC50)
        summary['MAE'].append(MAE)
        MSE = mean_squared_error(tar_df.pIC50, tar_df.predicted_pIC50)
        summary['MSE'].append(MSE)
        root_MSE = mean_squared_error(tar_df.pIC50, tar_df.predicted_pIC50, squared=False)
        summary['RMSE'].append(root_MSE)
        R2 = r2_score(tar_df.pIC50, tar_df.predicted_pIC50)
        summary['R2'].append(R2)

    for val in list(summary.keys())[1:]:
        for i,v in enumerate(summary[val]):
            summary[val][i] = float('%.4f'%v)
    summary_df = pd.DataFrame(summary)
    return summary_df


# In[ ]:




