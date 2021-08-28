#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_CVtraining(fold, params, save_model=False):
    cuda = True
    torch.manual_seed(42)
    random.seed(42)
    if cuda:
        torch.cuda.manual_seed(42)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 1000
    start = timeit.default_timer()
    
    print('Fold-'+str(fold))
    with open ('./X/cv/train_kfold-'+str(fold), 'rb') as train_fp:
        Train_X = pickle.load(train_fp)
    with open ('./X/internal-test', 'rb') as intest_fp:
        Intest_X = pickle.load(intest_fp)
    with open ('./X/cv/valid_kfold-'+str(fold), 'rb') as valid_fp:
        Valid_X = pickle.load(valid_fp)
    print('Training set: '+str(len(Train_X)))
    print('Internal-test set: '+str(len(Intest_X)))
    print('Validation set: '+str(len(Valid_X)))
    
    deg = torch.zeros(5, dtype=torch.long)
    for data in Train_X:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    batch = 16
    train_loader = DataLoader(Train_X, batch_size=batch, shuffle=True)
    intest_loader = DataLoader(Intest_X, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(Valid_X, batch_size=batch, shuffle=True)
    pred_loader = DataLoader(Valid_X, batch_size=1, shuffle=False, drop_last=False)
    print('Data has been loaded')
    
    Model = Net(npnalayers= params['npnalayers'],
                graphin = Train_X[0].x.shape[1],
                edgedim = Train_X[0].edge_attr.shape[1],
                predictnode= params['predictnode'], 
                nfinglayers= params['nfinglayers'], 
                fingdim=Train_X[0].fing.shape[1], 
                fingernode= params['fingernode'], 
                fingerout= params['fingerout'],
                dropout= params['dropout'], 
                mlpout= params['mlpout'], 
                mlpnode= params['mlpnode'],
                device=DEVICE,
                deg=deg
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
    
    target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
    Results = {'epoch':[],
               'loss_total_train':[], 'train_erbB4_MSE':[], 'train_egfr_MSE':[], 'train_met_MSE':[],
               'train_alk_MSE':[], 'train_erbB2_MSE':[], 'train_ret_MSE':[], 'train_ros1_MSE':[],
               'loss_total_internal-test':[],'internal-test_erbB4_MSE':[], 'internal-test_egfr_MSE':[], 'internal-test_met_MSE':[],
               'internal-test_alk_MSE':[], 'internal-test_erbB2_MSE':[], 'internal-test_ret_MSE':[], 'internal-test_ros1_MSE':[],
               'loss_total_valid':[],'valid_erbB4_MSE':[], 'valid_egfr_MSE':[], 'valid_met_MSE':[],
               'valid_alk_MSE':[], 'valid_erbB2_MSE':[], 'valid_ret_MSE':[], 'valid_ros1_MSE':[]
              }
    keys_list = list(Results)
    
    for epoch in range(EPOCHS):
        if epoch  % decay_interval == 0:
            Optimizer.param_groups[0]['lr'] *= lr_decay
        
        train_loss = eng.train(train_loader)
        intest_loss = eng.evaluate(intest_loader)
        Scheduler.step(intest_loss[0]) #total loss intest
        valid_loss = eng.evaluate(valid_loader)
        
        typeformat = []
        typeformat.append(epoch)
        typeformat.extend(train_loss)
        typeformat.extend(intest_loss)
        typeformat.extend(valid_loss)
        print('epoch:%d\ntrain total loss: %.3f\ntask1: %.3f, task2:%.3f, task3:%.3f ,task4: %.3f, task5:%.3f, task6:%.3f, task7:%.3f\ninternal-test total loss: %.3f\nloss1: %.3f, loss2: %.3f, loss3: %.3f , loss4: %.3f, loss5: %.3f, loss6: %.3f, loss7: %.3f\nvalid total loss: %.3f\nloss1: %.3f, loss2: %.3f, loss3: %.3f, loss4: %.3f, loss5: %.3f, loss6: %.3f, loss7: %.3f'
              %tuple(typeformat))
        
        for i in range(len(typeformat)):
            col = keys_list[i]
            if i == 0:
                Results[col].append(typeformat[i])
            else:
                Results[col].append(float('%.4f'%typeformat[i]))

        predicted_list = {'erbB4':[], 'egfr':[], 'met':[], 'alk':[], 'erbB2':[], 'ret':[], 'ros1':[]}
        prediction = Model.eval()
        for mol in pred_loader:
            predicted = prediction(mol)
            for t, v in enumerate(predicted):
                val = v.cpu().detach().numpy().tolist()
                pred_val = float('%.2f'%val[0][0])
                predicted_list[list(predicted_list.keys())[t]].append(pred_val)

        if epoch >= num_mean_vals:
            val_list = []
            for val in Results['loss_total_internal-test'][-num_mean_vals:]: #total loss intest
                val_list.append(val)
            avg = float('%.4f'%(np.mean(val_list)))
            if avg < best_loss and intest_loss[0] < best_loss:
                best_loss = avg
                best_valid = valid_loss
                early_stopping_counter = 0
                best_predicted = predicted_list
                if save_model:
                    Result_df = pd.DataFrame(Results)
                    Result_df.to_csv('./CV/MSE_result_cv'+str(fold)+'.csv')
                    with open('./CV/deg_cv'+str(fold), 'wb') as dg:
                        pickle.dump(deg, dg)
                    torch.save(Model.state_dict(), './CV/MT-PNA_cv'+str(fold)+'.pt')
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

