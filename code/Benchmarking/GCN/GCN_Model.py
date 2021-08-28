#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from deepchem.models import GraphConvModel, GCNModel
from deepchem.metrics import Metric, r2_score
from sklearn.metrics.regression import r2_score, mean_squared_error, mean_absolute_error

def define_regression_model(n_tasks=1, graph_conv_sizes=(128, 128), param = alk, config=default_config, model_dir='/tmp'):
       return GCNModel(n_tasks=n_tasks, graph_conv_layers=graph_conv_sizes,dropout=param['dropouts'],predictor_hidden_feats=param['dense_size'],
                       mode='regression',number_atom_features=30, batch_size=param['batch_size'],learning_rate=param['learning_rate'],
                       learning_rate_decay_time=param['learning_rate_decay_time'], optimizer_type='adam', configproto=config, model_dir=model_dir)
    


# In[ ]:


def train_and_validate(target, train, n_tasks, outdir, graph_conv_sizes, param, num_epochs, pickle_file_name, valid=None, test=None, gpu=None):
    if gpu is None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        config = tf1.ConfigProto(device_count={'GPU': 0, 'CPU': 1})
    else:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '%i' % gpu
        config = tf1.ConfigProto(
            gpu_options=tf1.GPUOptions(per_process_gpu_memory_fraction=0.75),
            device_count={'GPU': 1}
        )

    model_dir = op.join(outdir, 'model')
    ensure_dir(model_dir)
    model = define_regression_model(n_tasks, graph_conv_sizes=graph_conv_sizes,param=param, model_dir=model_dir, config=config)
  

    mse = Metric(mean_squared_error)

    best_loss = np.inf
    early_stopping_iter = 10
    num_mean_vals = 10
    early_stopping_counter = 0
    monitor_list = []

    for l in range(0, num_epochs):
        print('EPOCH %i' % l)
        model.fit(train, nb_epoch=1) 
        model.evaluate(train, [mse])
        if test is not None:
            try:
                model.evaluate(test, [mse])
                monitor = model.evaluate(test, [mse])
                monitor = monitor['mean_squared_error']
                monitor_list.append(monitor)
                if l >= num_mean_vals:
                  val_list = []
                  for val in monitor_list[-num_mean_vals:]:
                    val_list.append(val)
                  avg = float('%.4f'%(np.mean(val_list)))
                  if avg < best_loss and monitor < best_loss:
                    best_loss = avg
                    early_stopping_counter = 0
                    yhattrain = model.predict(train)
                    yhatvalid = model.predict(valid)
                    yhattest = model.predict(test)
                    eval_best_train = evaluate(train.y, yhattrain)
                    eval_best_valid = evaluate(valid.y, yhatvalid)
                    eval_best_test = evaluate(test.y, yhattest)
                  else:
                   early_stopping_counter += 1
                   print("Count = ", early_stopping_counter)

                if early_stopping_counter == early_stopping_iter:
                  yhat = [item for sublist in yhatvalid for item in sublist]
                  ytrue = [item for sublist in valid.y for item in sublist]
                    
                  train_yhat = [item for sublist in yhattrain for item in sublist]
                  train_ytrue = [item for sublist in train.y for item in sublist]
                    
                  result = {'ytrue':ytrue,'yhat':yhat}
                  result_df = pd.DataFrame(result)
                  save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
                  result_df.to_csv(save_dir+target+'/dataframe/cv/Result/Extest_fulltrain_GCN.csv', index=False)
                 
                  result = {'ytrue':train_ytrue,'yhat':train_yhat}
                  result_df = pd.DataFrame(result)
                  save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
                  result_df.to_csv(save_dir+target+'/dataframe/cv/Result/Train_fulltrain_GCN.csv', index=False)
                    
                  print('Early stopping')
                  print("Best Value")
                  print('Train r2, mse, mae', eval_best_train)
                  print('Extest r2, mse, mae',eval_best_valid)
                  print("---------------------------------------------")
                  return eval_best_train,eval_best_valid
                  break
            except TypeError: 
                print('No validation performance available')
    else:
        return train.y, yhattrain, train.w

