#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold


# In[92]:


target_list = ['erbB4', 'egfr', 'met', 'alk', 'erbB2', 'ret', 'ros1']

def splitData(df, split_size, label_column):
    train, test = train_test_split(df, test_size=split_size, shuffle=True, stratify= df.loc[:,label_column])
    return train, test

def count_taskDataset(df, target_list):
    tk_df = pd.DataFrame()
    result = pd.DataFrame(index=[0], columns=target_list)
    result['total'] = df.shape[0]
    print('total data: '+str(df.shape[0]))
    for tar in target_list:
        tar_df = df.dropna(subset=['pIC50_'+tar])
        tar_df = tar_df[['SMILES_NS','pIC50_'+tar]]#, 'dataset']]
        tar_df.rename(columns={'pIC50_'+tar:'pIC50'},inplace=True)#, 'predicted_pIC50_'+tar: 'predicted_pIC50'},inplace=True)
        tar_df['target'] = tar
        tk_df = pd.concat([tk_df, tar_df],axis=0)
        result[tar] = tar_df.shape[0]
        print(tar+' data: '+str(tar_df.shape[0]))
    tk_df.sort_index(inplace=True)
    return result, tk_df

def defineLabel(df, target_list):
    count_dict = {'erbB4':0, 'egfr':0, 'met':0, 'alk':0, 'erbB2':0, 'ret':0, 'ros1':0}
    co = [0]*7
    for i, tar in(enumerate(target_list)):
        for v in range(len(df['SMILES_NS'])):
            if np.isnan(df['pIC50_'+tar][v]) == False:
                count_dict[tar] += 1
    target_order = sorted(count_dict, key=count_dict.get)
    co = [0]*7
    for i, tar in(enumerate(target_order)):
        for v in range(len(df['SMILES_NS'])):
            if np.isnan(df['pIC50_'+tar][v]) == False:
                co[i] += 1
    Label = []
    for index, row in df.iterrows():
        for i, tar in(enumerate(target_order)):
            if np.isnan(row['pIC50_'+tar]) == False:
                Label.append(i+1)
                break
    return Label

def extractSmiles(df, target_list):
    smi_dict = {}
    for tar in target_list:
        tar_df = df[df.target == tar]
        tar_smi = tar_df.SMILES_NS.values
        smi_dict[tar] = tar_smi
    return smi_dict

def gen2Dmols(smi_dict, target_list):
    from rdkit import Chem
    mol_dict = {}
    for tar in target_list:
        smiles = smi_dict.get(tar)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        mol_dict[tar] = mols
    return mol_dict

def genMorganFP(mol_dict, rad, target_list):
    from rdkit.Chem import AllChem
    fp_dict = {}
    for tar in target_list:
        fp_dict[tar] = []
        for m in mol_dict.get(tar):
            fp = AllChem.GetMorganFingerprint(m, rad)
            fp_dict[tar].append(fp)
    return fp_dict

def similarityTanimoto(fp_dict_col, fp_dict_ind, target_list, data):
    from rdkit import DataStructs
    sim_df_dict = {}
    for tar in target_list:
        col_name = []
        for ind in range(len(fp_dict_col.get(tar))):
            col = ind
            col_name.append(col)
        index_name = []
        for ind in range(len(fp_dict_ind.get(tar))):
            index = ind
            index_name.append(index)
        sim_df = pd.DataFrame(index=index_name, columns=col_name)
        for i in range(0,len(fp_dict_ind.get(tar))):
            test_fp = fp_dict_ind[tar][i]
            for j in range(0,len(fp_dict_col.get(tar))):
                if j > i:
                    train_fp = fp_dict_col[tar][j]
                    sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
                    sim_df.iloc[i,j] = float('%.4f'%sim)
        sim_df_dict[tar] = sim_df
        print(tar+' similarity metrics: '+str(sim_df_dict[tar].shape))
    return sim_df_dict

def sub_activityCliff(sim_df, tar_df):
    ac_dict = {'ID_1':[],'ID_2':[],'SMILES_1':[],'SMILES_2':[],'structure_similarity':[]}
    for i in sim_df.index:
        print(tar+': 1_index-'+str(i)+'/'+str(len(sim_df.index)-1))
        id_df = sim_df.iloc[i]
        id_df.dropna(inplace=True)
        ac_dict['ID_1'].extend([i]*id_df.shape[0])
        ac_dict['SMILES_1'].extend([tar_df.SMILES_NS.iloc[i]]*id_df.shape[0])
        for j in id_df.index:
            j = int(j)
            ac_dict['ID_2'].append(j)
            ac_dict['SMILES_2'].append(tar_df.SMILES_NS.iloc[j])
            ac_dict['structure_similarity'].append(id_df[j])
    ac_df = pd.DataFrame(ac_dict)
    tar_smi_1 = tar_df[['SMILES_NS', 'pIC50']].rename(columns={'SMILES_NS':'SMILES_1'})
    tar_smi_2 = tar_smi_1.rename(columns={'SMILES_1':'SMILES_2'})
    ac_df = pd.merge(ac_df, tar_smi_1, on=['SMILES_1'], how='inner').rename(columns={'pIC50':'pIC50_1'})
    ac_df = pd.merge(ac_df, tar_smi_2, on=['SMILES_2'], how='inner').rename(columns={'pIC50':'pIC50_2'})
    ac_df.sort_values(by=['ID_1','ID_2'], inplace=True)
    pic50_range = max(ac_df.pIC50_1)-min(ac_df.pIC50_1)
    ac_df['delta_pIC50'] = np.nan
    ac_df['activity_similarity'] = np.nan
    ac_df['SAS_map'] = np.nan
    for i in range(len(ac_df.index)):
        print(tar+': 2_index-'+str(i)+'/'+str(len(ac_df.index)-1))
        ac_df['delta_pIC50'].iloc[i] = abs(ac_df.pIC50_1.iloc[i]-ac_df.pIC50_2.iloc[i])
        ac_df['activity_similarity'].iloc[i] = float('%.4f'%(1-(abs(ac_df.pIC50_1.iloc[i]-ac_df.pIC50_2.iloc[i])/pic50_range)))
        if float('%.2f'%(ac_df.structure_similarity.iloc[i])) >= 0.55:
            if ac_df.delta_pIC50.iloc[i] >= 2:
                ac_df['SAS_map'].iloc[i] = 'activity_cliffs'
            else:
                ac_df['SAS_map'].iloc[i] = 'similarity_cliffs'
        else:
            if ac_df.delta_pIC50.iloc[i] >= 2:
                ac_df['SAS_map'].iloc[i] = 'non_descriptive'
            else:
                ac_df['SAS_map'].iloc[i] = 'smooth_SAR' 
    print(tar+' SAS map: '+str(ac_df.shape))
    print(ac_df.SAS_map.value_counts())
    return ac_df


# In[93]:


raw_df = pd.read_csv('./7TKs_ic50_chembl+bindingdb.csv', index_col=0)
print(raw_df.shape)

Label = defineLabel(raw_df, target_list)
raw_df['Label'] = Label


# In[94]:


tv, test = splitData(raw_df, 0.1, 'Label')
tv.reset_index(drop=True, inplace=True)
print(tv.shape)

_, tk_tv = count_taskDataset(tv, target_list)
tk_tv.reset_index(drop=True, inplace=True)
tk_tv.to_csv('./7TK_train-intest.csv')


# In[84]:


tv_smi_dict = extractSmiles(tk_tv, target_list)
tv_mol_dict = gen2Dmols(tv_smi_dict, target_list)
tv_fp_dict = genMorganFP(tv_mol_dict, 4, target_list)
tv_sim_dict = similarityTanimoto(tv_fp_dict, tv_fp_dict, target_list, 'train')
for tar in target_list:
    tar_pro = tk_tv[tk_tv.target == tar].reset_index(drop=True)
    sim_df = tv_sim_dict[tar]
    tar_ac_df = sub_activityCliff(tv_sim_dict[tar], tar_pro.copy())
    
    ac = tar_ac_df[tar_ac_df.SAS_map == 'activity_cliffs']
    ac_smiles = ac.SMILES_1.values.tolist()
    ac_smiles.extend(ac.SMILES_2.values.tolist())
    ac_smiles = list(set(ac_smiles))
    tar_tv = tk_tv[tk_tv.target==tar].reset_index(drop=True)
    tar_rm_df = tar_tv[~tar_tv.SMILES_NS.isin(ac_smiles)]
    tar_rm_df.reset_index(drop=True, inplace=True)
    print(tar,'after AC analysis',tar_rm_df.shape)
    tar_rm_df.to_csv('./'+tar+'_train-intest.csv')


# In[ ]:


for t, tar in enumerate(target_list):
    tar_df = pd.read_csv('./'+tar+'_train-intest.csv', index_col=0)
    tar_df.rename(columns={'pIC50':'pIC50_'+tar},inplace=True)
    tar_df.drop(columns=['target'], inplace=True)
    print(tar_df.shape)
    if t == 0:
        merge_df = tar_df
    else:
        merge_df = pd.merge(merge_df, tar_df, on=['SMILES_NS'], how='outer')
print(merge_df.shape)
merge_df.to_csv('./train-intest_after_AC_removal.csv')

