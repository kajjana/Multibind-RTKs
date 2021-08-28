#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_CVtraining(fold,target,param):
    save_dir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task/'
    target = str(target)
    
    start = timeit.default_timer()
    
    print('Fold-'+str(fold))
    train_csv = save_dir+target+'/dataframe/cv/train_kfold-'+str(fold)+'.csv'
    train = pd.read_csv(train_csv).sample(frac=1)

    intest_csv = save_dir+target+'/dataframe/internal-test.csv'
    intest = pd.read_csv(intest_csv)

    valid_csv = save_dir+target+'/dataframe/cv/valid_kfold-'+str(fold)+'.csv'
    valid = pd.read_csv(valid_csv)
    
    print('Training set: '+str(len(train)))
    print('Internal-test set: '+str(len(intest)))
    print('Validation set: '+str(len(valid)))

    train = loadData(train)
    intest = loadData(intest)
    valid = loadData(valid)
    print('Data has been loaded')
    
    
    train, valid =  train_and_validate(fold,target,train = train, n_tasks=1, outdir = '/share/galaxy/fahsai/NSCLC/Sin/single_task/single_task',
                                       graph_conv_sizes = (128, 128), param = param, num_epochs = 3000,pickle_file_name='/content/test2.pkl', 
                                       valid=valid, test=intest, gpu=0)
    return train, valid

