#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna

def objective(trial):
    params = {
        'npnalayers': trial.suggest_int('npnalayers', 1, 8),
        'predictnode':trial.suggest_int('predictnode', 16, 1024),
        'nfinglayers': trial.suggest_int('nfinglayers', 1, 8),
        'fingernode':trial.suggest_int('fingernode', 16, 1024),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.8),
        'fingerout':trial.suggest_int('fingerout', 16, 1024),
        'mlpout':trial.suggest_int('mlpout', 16, 1024),
        'mlpnode':trial.suggest_int('mlpnode', 16, 1024),
        'lr':trial.suggest_loguniform('lr', 1e-5, 1e-3),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3),
        'batch':trial.suggest_int('batch', 4, 256)
    }
    target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
    target = target_list[0] #erbB4
    print(target+' tuning')
    all_losses = []
    try:
        for f_ in range(10):
                temp_loss,_ = run_CVtraining(target, f_, params, save_model=False)
                all_losses.append(temp_loss) #optimize total valid loss
    except:
        print('This parameter set is not matched')
        all_losses = np.inf
    return np.mean(all_losses)


# In[ ]:


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


# In[ ]:


target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']
target = target_list[0] #erbB4
print(target)

print('best trial:')
trial_ = study.best_trial

print(trial_.values)
print(trial_.params)

