#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from deepchem.models import GATModel

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

def define_regression_model(n_tasks=1, graph_attention_layers = [8,8],n_attention_heads=8, batch_size=128,predictor_hidden_feats = 128,predictor_dropout = 0.2,
                               learning_rate_decay_time = 1000, learning_rate=0.001, config=default_config, model_dir='/tmp'):
       return GATModel(n_tasks=n_tasks, graph_attention_layers=graph_attention_layers, n_attention_heads=n_attention_heads, 
                               predictor_hidden_feats =  predictor_hidden_feats, predictor_dropout = predictor_dropout, number_atom_features = 30, batch_size=batch_size,learning_rate=learning_rate,
                               learning_rate_decay_time=learning_rate_decay_time, optimizer_type='adam', configproto=config, model_dir=model_dir)
    
train, intest = CreateDataset(target='ros1')


# In[ ]:


#Bayesian Search
params_dict = {
    'n_attention_heads':8,
    'predictor_dropout': 0.1,
    'predictor_hidden_feats':128,
    'learning_rate_decay_time':1000,
    'batch_size': 32,
    'learning_rate': 0.0001}

search_range = {
    'n_attention_heads':4,
    'predictor_dropout': 8,
    'predictor_hidden_feats':8,
    'learning_rate_decay_time':8,
    'batch_size': 8,
    'learning_rate': 10}

optimizer = dc.hyper.GaussianProcessHyperparamOpt(define_regression_model)
mse = Metric(mean_squared_error)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict, train, intest, metric=mse,nb_epoch=50,use_max=False,max_iter=50)

#printing out results
print("\n===================BEST MODEL=================")
print(best_model)
print("\n===================BEST Params=================")
print(best_hyperparams)
print("\n===================ALL_RESULTS=================")
print(all_results)

