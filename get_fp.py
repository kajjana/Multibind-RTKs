#!/usr/bin/env python
# coding: utf-8



import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_notebook
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Avalon import pyAvalonTools
from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
import pybel



#define fingerprint generating fuctions

if not isJVMStarted():
    cdk_path = './cdk-2.3.jar'
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
    cdk = JPackage('org').openscience.cdk

def cdk_parser_smiles(smi):
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    try:
        mol = sp.parseSmiles(smi)
    except:
        raise IOError('invalid smiles input')
    return mol

def cdk_fingerprint(smi, fp_type="standard", size=2048, depth=6): #'bit'
    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860
    else:
        nbit = size
        
    _fingerprinters = {"daylight":cdk.fingerprint.Fingerprinter(size, depth)
                     , "extended":cdk.fingerprint.ExtendedFingerprinter(size, depth)
                     , "graph":cdk.fingerprint.GraphOnlyFingerprinter(size, depth)
                     , "maccs":cdk.fingerprint.MACCSFingerprinter()
                     , "pubchem":cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance())
                     , "estate":cdk.fingerprint.EStateFingerprinter()
                     , "hybridization":cdk.fingerprint.HybridizationFingerprinter(size, depth)
                     , "lingo":cdk.fingerprint.LingoFingerprinter(depth)
                     , "klekota_roth":cdk.fingerprint.KlekotaRothFingerprinter()
                     , "shortestpath":cdk.fingerprint.ShortestPathFingerprinter(size)
                     , "signature": cdk.fingerprint.SignatureFingerprinter(depth)
                     , "circular": cdk.fingerprint.CircularFingerprinter()
                     , "AtomPair": cdk.fingerprint.AtomPairs2DFingerprinter()
                     }
    
    mol = cdk_parser_smiles(smi)
    if fp_type in _fingerprinters:
        fingerprinter = _fingerprinters[fp_type]
    else:
        raise IOError('invalid fingerprint type')
        
    fp = fingerprinter.getBitFingerprint(mol).asBitSet()
    bits = []
    idx = fp.nextSetBit(0)
    while idx >= 0:
        bits.append(idx)
        idx = fp.nextSetBit(idx + 1)

    return bits

def rdk_fingerprint(smi, fp_type="rdkit", size=2048):
    _fingerprinters = {"rdkit": Chem.rdmolops.RDKFingerprint
                     , "maccs": MACCSkeys.GenMACCSKeys
                     , "TopologicalTorsion": Torsions.GetTopologicalTorsionFingerprint
                     , "Avalon": pyAvalonTools.GetAvalonFP}
    mol = Chem.MolFromSmiles(smi)
    if fp_type in _fingerprinters:
        fingerprinter = _fingerprinters[fp_type]
        fp = fingerprinter(mol)
    elif fp_type == "AtomPair":
        fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
    elif fp_type == "Morgan":
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    else:
        raise IOError('invalid fingerprint type')
    if fp_type == "AtomPair":
        res = np.array(fp)
    else:
        temp = fp.GetOnBits()
        res = [i for i in temp]
    
    return res

def ob_fingerprint(smi, fp_type='FP2', nbit=307, output='bit'):
    mol = pybel.readstring("smi", smi)
    if fp_type == 'FP2':
        fp = mol.calcfp('FP2')
    elif fp_type == 'FP3':
        fp = mol.calcfp('FP3')
    elif fp_type == 'FP4':
        fp = mol.calcfp('FP4')
    bits = fp.bits
    bits = [x for x in bits if x < nbit]
    if output == 'bit':
        return bits
    else:
        vec = np.zeros(nbit)
        vec[bits] = 1
        vec = vec.astype(int)
        return vec

def get_fingerprint(smi, fp_type, nbit=None, depth=None): #'bit'
    if fp_type in ["daylight", 
                   "extended", 
                   "graph", 
                   "pubchem", 
                   "estate", 
                   "hybridization", 
                   "lingo", 
                   "klekota_roth", 
                   "shortestpath", 
                   "signature", 
                   "circular",
                   "AtomPair"]:
        if nbit is None:
            nbit = 1024
        if depth is None:
            depth = 6 
        res = cdk_fingerprint(smi, fp_type, nbit, depth)
        
    elif fp_type in ["rdkit", 
                     "maccs", 
                     "TopologicalTorsion", 
                     "Avalon"]:
        res = rdk_fingerprint(smi, fp_type, nbit)
        
    elif fp_type in ["FP2", 
                     "FP3", 
                     "FP4"]:
        if nbit is None:
            nbit = 307
        res = ob_fingerprint(smi, fp_type, nbit)
        
    else:
        raise IOError('invalid fingerprint type')
    return res

def BitVect_to_NumpyArray(fp, fp_type):
    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota_roth':
        nbit = 4860
    elif fp_type == 'AtomPair':
        nbit = 2048
    elif fp_type == ['FP2','FP3','FP4']:
        nbit = 307
    else:
        nbit = 2048
        
    bitvect = [0]*nbit
    for val in fp:
        bitvect[val-1] = 1
        
    return np.array(list(bitvect))

def feature_engineer(df, column_name, prefix):
    features = []
    for row in tqdm(df.index, total=len(df.index)):
        s = df.loc[row, column_name]
        feature = []
        for char in s:
            feature.append(str(char))
        features.append(feature)

    data = pd.DataFrame(features, 
                        columns=[prefix + str(i) for i in range(len(feature))])
    return data


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('dataset_FP', type=str)
args = parser.parse_args()

csv = args.dataset
save = args.dataset_FP

#import smiles dataframe
ligand = pd.read_csv(csv, index_col=0)
df = ligand[['smiles']].reset_index(drop=True)
print(df.shape)
df.head()

tqdm.pandas()
data_df = ligand[['smiles']]
all_fp_types = ["daylight",
                "extended", 
                "graph", 
                "pubchem", 
                "estate", 
                "hybridization", 
                "lingo", 
                "klekota_roth", 
                "circular",
                "rdkit", 
                "maccs",
                "AtomPair",
                "Avalon",
                "FP2",
                "FP3",
                "FP4"]

for fp in all_fp_types:
    print(fp)
    sub_df = df.copy()
    
    print("features are in generating process")
    fp_data = sub_df.progress_apply(lambda row: BitVect_to_NumpyArray(get_fingerprint(row['smiles'], fp_type=fp),fp), axis=1)
    sub_df.loc[sub_df.index,fp] = [str(list(fp_data[i])).strip('[]') for i in range(len(fp_data.index))]
    sub_df.loc[sub_df.index,fp] = sub_df.loc[sub_df.index,fp].astype(str).apply(lambda x: x.replace(" ", ""))
    sub_df.loc[sub_df.index,fp] = sub_df.loc[sub_df.index,fp].astype(str).apply(lambda x: x.replace("\n", ""))
    sub_df.loc[sub_df.index,fp] = sub_df.loc[sub_df.index,fp].astype(str).apply(lambda x: x.replace(",", ""))
    
    print("features are in engineering process")
    fp_data_eng = feature_engineer(sub_df, column_name=fp, prefix=fp+'FP')
    fp_df = fp_data_eng.apply(pd.to_numeric, errors='ignore')   
    #fp_df.to_csv('./Fingerprints/' + fp + 'Fingerprint.csv',index=False)
    data_df = pd.concat([data_df, fp_df], axis=1)

All = data_df
All.to_csv(save)
print("Fingerprint Saved")
print('\n')
print('Finished')





