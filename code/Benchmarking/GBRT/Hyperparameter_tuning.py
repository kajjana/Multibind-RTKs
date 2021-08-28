#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# In[ ]:


space  = {'max_depth':Integer(2, 20),
          'learning_rate':Real(0.001, 0.1, "uniform"),
          'min_samples_split':Integer(2, 5),
          'min_samples_leaf':Integer(2, 5),
          'max_features': Categorical(['sqrt','auto','log2']),
          'n_estimators': Integer(100, 1000)
          }
bsearch = BayesSearchCV(estimator = GradientBoostingRegressor(random_state=42), 
search_spaces = space, scoring='neg_mean_squared_error',n_jobs=1, n_iter=50,iid=False, cv=10)
bsearch.fit(x_train,y_train)
print(bsearch.best_score_)
print(bsearch.best_params_)

