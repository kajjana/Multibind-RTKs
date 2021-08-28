#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import deepchem as dc
import pickle
import tempfile

def loadData(df):
    with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
      df.to_csv(tmpfile.name)
      loader = dc.data.CSVLoader(["pIC50"], feature_field="SMILES_NS",
                             featurizer=dc.feat.MolGraphConvFeaturizer())
      data = loader.create_dataset(tmpfile.name)
    return data


# In[ ]:


def CreateDataset(target):
    save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
    target = str(target)
    
    train_csv = save_dir+target+'/train.csv'
    train = pd.read_csv(train_csv).dropna()
    intest_csv = save_dir+target+'/internal-test.csv'
    intest = pd.read_csv(intest_csv).dropna()

    print('Training set: '+str(len(train)))
    print('Internal-test set: '+str(len(intest)))

    train = loadData(train)
    intest = loadData(intest)
    print('Data has been loaded')
    
    return train, intest

def define_regression_model(n_tasks=1, graph_conv_sizes=(128, 128), dense_size=256, batch_size=128,dropouts=0.0,learning_rate_decay_time=1000,
                               learning_rate=dc.models.optimizers.ExponentialDecay(0.001, 0.9, 1000), config=default_config, model_dir='/tmp'):
       return GCNModel(n_tasks=n_tasks, graph_conv_layers=graph_conv_sizes,dropout=0.0,predictor_hidden_feats=dense_size,
                       mode='regression',number_atom_features=30, batch_size=batch_size,learning_rate=learning_rate,
                       learning_rate_decay_time=learning_rate_decay_time, optimizer_type='adam', configproto=config, model_dir=model_dir)

train, intest = CreateDataset(target='ros1')


# In[ ]:


#Bayesian Search
params_dict = {
    'dense_size': 128,
    'dropouts': 0.1,
    'batch_size': 32,
    'learning_rate_decay_time':1000,
    'learning_rate': 0.0001}
search_range = {
    'dense_size': 8,
    'dropouts': 8,
    'batch_size': 8,
    'learning_rate_decay_time':5,
    'learning_rate': 10}


optimizer = dc.hyper.GaussianProcessHyperparamOpt(define_regression_model)
mse = Metric(mean_squared_error)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict, train, intest, search_range = search_range, 
                                                                        metric=mse,nb_epoch=100,use_max=False,max_iter=50)

#printing out results
print("\n===================BEST MODEL=================")
print(best_model)
print("\n===================BEST Params=================")
print(best_hyperparams)
print("\n===================ALL_RESULTS=================")
print(all_results)

