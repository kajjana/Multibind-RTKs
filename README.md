# Multibind-RTKs
## Prerequisites

Prerequisite libraries are listed 
1) scikit-learn
2) rdkit
3) Pytorch
4) Pytorch-Geometric
5) jpype1
6) openbabel

```
pip install scikit-learn
conda install -y -c rdkit rdkit
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric
conda install -y -c conda-forge jpype1=0.7.5
conda install -y -c openbabel openbabel=2.4.1
```
## Usage
### For Screening Purpose
Prepare the csv file containing index and 'smiles' column of screening molecules. The smiles must have desalted via data preprocessing processes. In this case, we use OTAVA.csv as a sample dataset.

The structure of the `root_dir` should be:
```
root_dir
├── get_fp.py
├── featurized_screen.py
├── predict-ad.py
├── OTAVA.csv (screening dataset)
├── cdk-2.3.jar
│ 
├── PCA_FP
├── X
├── Model
│   ├── prepca.model 
│   ├── deg_pretrain.pkl  
│   └── pretrain.model
├── AD
│   ├── train_for_AD.csv 
│   └── CVprediction_for_AD.csv
└── Results
```
Generate 16 Fingerprints from ```OTAVA.csv``` , then concatenate and save it to ```OTAVA_FP.csv```. 

```bash
python get_fp.py OTAVA.csv OTAVA_FP.csv
```

Select dataset file and its fingerprint file, pca model, and assign task as 'Screen'. The dimensional reduction via PCA will perform on fingerprint by ```prepca.model```.
The graph and pca fingerprint feature of the selected dataset will be loaded to ```OTAVA.pkl```.
The ```OTAVA_PCA16FPs.csv``` will be collected in ```PCA_FP``` folder and ```OTAVA.pkl``` will be collected in ```X``` folder.

```bash
python featurized.py OTAVA.csv OTAVA_FP.csv prepca.model Screen
```
Use ```pretrain.model``` to predict pIC50 of selected dataset .

```bash
python predict.py pretrain.model OTAVA.pkl AD
```
The ```Result_OTAVA.csv``` that contain the predicted values will be collected in ```Results``` folder. The applicability domain (AD) analysis will be performed automatically.
 
## For custom model training purpose
Prepare the csv file containing 9 columns consist of index, 'smiles', 'pIC50_erbB4',	'pIC50_egfr',	'pIC50_met',	'pIC50_alk',	'pIC50_erbB2',	'pIC50_ret', and	'pIC50_ros1'.
of molecules respectively. The smiles must have desalted via data preprocessing processes.
The structure of the `root_dir` should be:
```
root_dir
├── get_fp.py
├── PCA.py
├── featurized_screen.py
├── train.py
├── predict-ad.py
├── train.csv (training set)
├── valid.csv (validation set)
├── cdk-2.3.jar
│ 
├── PCA_FP
├── X
├── Model
└── Results
```

After run the ```get_fp.py``` on ```train.csv``` and ```valid.csv```, select the training set and its fingerprint file then train the pca model with the training set and save the trained pca model to ```pca.model```. The model will be collected in ```Model``` folder.

```bash
python PCA.py train.csv train_FP.csv pca.model
```

Run ```featurized.py``` with ```pca.model``` and assign task as 'Train' using training and validation set to generate its feature .pkl file 

```bash
python featurized.py train.csv train_FP.csv pca.model Train
```

Use the ```train.pkl``` and ```valid.pkl``` to train the model and save the custom model to ```model.model```. The model and ```deg_model.pkl``` will be saved in ```Model``` folder and ```model_MSE_result.csv``` will be saved in ```Results``` folder. 

```bash
python train_model.py train.pkl valid.pkl model.model
```
Run ```predict.py``` but use the custom model instead of the provided pretrain model to predict pIC50 of selected dataset. The AD analysis does not support the custom model.

```bash
python predict.py model.model OTAVA.pkl noAD
```

