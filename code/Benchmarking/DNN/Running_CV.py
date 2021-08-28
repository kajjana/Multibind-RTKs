#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_CVtraining(target, fold, params, save_model=False):
    cuda = True
    torch.manual_seed(42)
    random.seed(42)
    if cuda:
        torch.cuda.manual_seed(42)
    
    model_dir = '/'+target
    data_dir = '/'+target
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 1000
    start = timeit.default_timer()
    
    print('Fold-'+str(fold))
    with open (data_dir+'/X/cv/train_kfold-'+str(fold), 'rb') as train_fp:
        Train_X = pickle.load(train_fp)
    with open (data_dir+'/X/internal-test', 'rb') as intest_fp:
        Intest_X = pickle.load(intest_fp)
    with open (data_dir+'/X/cv/valid_kfold-'+str(fold), 'rb') as valid_fp:
        Valid_X = pickle.load(valid_fp)
    print('Training set: '+str(len(Train_X)))
    print('Internal-test set: '+str(len(Intest_X)))
    print('Validation set: '+str(len(Valid_X)))
    
    batch = params['batch']
    if len(Train_X)%batch == 1:
        train_loader = DataLoader(Train_X, batch_size=batch, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(Train_X, batch_size=batch, shuffle=True)
    intest_loader = DataLoader(Intest_X, batch_size=batch, shuffle=True, drop_last=False)
    valid_loader = DataLoader(Valid_X, batch_size=batch, shuffle=True, drop_last=False)
    pred_loader = DataLoader(Valid_X, batch_size=1, shuffle=False, drop_last=False)
    print('Data has been loaded')
    
    Model = Net(nfinglayers= params['nfinglayers'], 
                fingdim=Train_X[0].fing.shape[1],
                fingernode= params['fingernode'], 
                fingerout= params['fingerout'],
                dropout= params['dropout'], 
                device=DEVICE,
                 ).to(DEVICE)
    Optimizer = optim.Adam(Model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    Scheduler = ReduceLROnPlateau(Optimizer, mode='min', factor=0.5, patience=20,
                              min_lr=0.00001)
    
    eng = Engine(model=Model, optimizer=Optimizer, device=DEVICE)
    
    best_loss = np.inf
    best_valid = [np.inf]*8
    early_stopping_iter = 10
    num_mean_vals = 10
    early_stopping_counter = 0
    decay_interval = 10
    lr_decay = 0.5
    
    Results = {'epoch':[],
               'loss_train':[],
               'loss_internal-test':[],
               'loss_valid':[]
              }
    keys_list = list(Results)
    
    for epoch in range(EPOCHS):
        if epoch  % decay_interval == 0:
            Optimizer.param_groups[0]['lr'] *= lr_decay
        
        train_loss = eng.train(train_loader)
        intest_loss = eng.evaluate(intest_loader)
        valid_loss = eng.evaluate(valid_loader)

        typeformat = []
        typeformat.append(epoch)
        typeformat.append(train_loss)
        typeformat.append(intest_loss)
        typeformat.append(valid_loss)
        print('epoch:%d\ntrain loss: %.3f\ninternal-test loss: %.3f\nvalid loss: %.3f'
              %tuple(typeformat))
        
        for i in range(len(typeformat)):
            col = keys_list[i]
            if i == 0:
                Results[col].append(typeformat[i])
            else:
                Results[col].append(float('%.4f'%typeformat[i]))
        
        predicted_list = []
        prediction = Model.eval()
        for mol in pred_loader:
            predicted = prediction(mol)
            pred = predicted.cpu().detach().numpy().tolist()
            predicted_list.append(float('%.2f'%pred[0][0]))

        if epoch >= num_mean_vals:
            val_list = []
            for val in Results['loss_internal-test'][-num_mean_vals:]: #total loss intest
                val_list.append(val)
            avg = float('%.4f'%(np.mean(val_list)))
            if avg < best_loss and intest_loss < best_loss:
                best_loss = avg
                best_valid = valid_loss
                best_predicted = predicted_list
                early_stopping_counter = 0
                if save_model:
                    Result_df = pd.DataFrame(Results)
                    Result_df.to_csv(model_dir+'/results/CV/MSE_result_cv'+str(fold)+'.csv')
                    torch.save(Model.state_dict(), model_dir+'/trained_model/CV/ST-DNN_cv'+str(fold)+'.pt')
                    print('Saved Model-'+str(fold))
            else:
                early_stopping_counter += 1
        
        print('current best loss = '+str(best_loss))
        print('current best valid loss = '+str(best_valid))
        print('counter = '+str(early_stopping_counter))
        
        if early_stopping_counter == early_stopping_iter:
            print('Early stopping')
            break
    
    end = timeit.default_timer()
    time = end - start
    print('Fold-'+str(fold)+' has finished in '+ str(float('%.4f'%(time/60)))+' mins')
    return best_valid, best_predicted

