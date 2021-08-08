# Multibind-RTKs
## Prerequisites
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
### For Screening Only
Prepare the csv file containing only 'smiles' column of screening molecules. The smiles must have desalted via data preprocessing processes. In this case, we use OTAVA.csv as a sample dataset.

The structure of the `root_dir` should be:
```
root_dir
├── get_fp.py
├── featurized_screen.py
├── predict.py
├── OTAVA.csv (screening dataset)
├── cdk-2.3.jar
├── Fingerprints
├── PCA_FP
├── X
├── Model
│   ├── prepca.model 
│   ├── deg_pretrain.pkl  
│   └── pretrain.model
└── Results
```
Generate 16 Fingerprints from ```OTAVA.csv``` , then concatenate and save it to ```OTAVA_FP.csv```. 16 fingerprints csv files will be collected in ```Fingerprints``` folder separately.

```bash
python get_fp.py OTAVA.csv OTAVA_FP.csv
```

Select dataset file and its fingerprint file. The dimensional reduction via PCA will perform on fingerprint by ```prepca.model```.
The graph and pca fingerprint feature of the selected dataset will be loaded to ```dataset.pkl```.
The ```dataset_PCA16FPs.csv``` will be collected in ```PCA_FP``` folder and ```dataset.pkl``` will be collected in ```X``` folder.

```bash
python featurized_screen.py OTAVA.csv OTAVA_FP.csv prepca.model
```
Use ```pretrain.model``` to predict pIC50 of selected dataset .

```bash
python predict.py pretrain.model OTAVA.pkl
```
The ```Result_OTAVA.csv``` that contain the predicted values will be collected in ```Results``` folder.
 
## Trained the custom model
After run the ```get_fp.py``` on ```train.csv``` and ```test.csv```, select the train dataset and its fingerprint file then train the pca model on the train dataset and save the trained pca model to ```pca.model```. The model will be collected in ```Model``` folder.

```bash
python PCA.py train.csv train_FP.csv pca.model
```

Run ```featurized_training.py``` with ```pca.model``` on train and test dataset to generate its feature .pkl file 

Use the ```train.pkl``` and ```test.pkl``` to train the model and save the custom model to ```model.model```. The model and ```deg_model.pkl``` will be saved in ```Model``` folder and ```model_MSE_result.csv``` will be saved in ```Results``` folder. 

```bash
python train_model.py train.pkl test.pkl model.model
```
Run ```predict.py``` but use the custom model instead of the provided pretrain model to predict pIC50 of selected dataset.

