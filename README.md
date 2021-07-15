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
The structure of the `root_dir` should be:
```
root_dir
├── dataset.csv
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
Generate 16 Fingerprints from ```dataset.csv``` , then concatenate and save it to ```dataset_FP.csv```. 16 fingerprints csv files will be collected in ```Fingerprints``` folder separately.

```bash
python GH_Fingerprint_Generation.py dataset.csv dataset_FP.csv
```

Select dataset file and its fingerprint file. The dimensional reduction via PCA will perform on fingerprint by ```prepca.model```.
The graph and pca fingerprint feature of the selected dataset will be loaded to ```dataset.pkl```.
The ```dataset_PCA16FPs.csv``` will be collected in ```PCA_FP``` folder and ```dataset.pkl``` will be collected in ```X``` folder.

```bash
python GH_Feature_Generation.py dataset.csv dataset_FP.csv prepca.model
```
Use ```pretrain.model``` to predict pIC50 of selected dataset .

```bash
python GH_Predict.py pretrain.model dataset.pkl
```
The ```Result_dataset.csv``` that contain the predicted values will be collected in ```Results``` folder.
 
## Trained the custom model
After run the ```GH_Fingerprint_Generation.py``` on ```train.csv``` and ```test.csv```, select the train dataset and its fingerprint file then train the pca model on the train dataset and save the trained pca model to ```pca.model```. The model will be collected in ```Model``` folder.

```bash
python GH_PCA.py train.csv train_FP.csv pca.model
```

Run ```GH_Feature_Generation.py``` with ```pca.model``` on train and test dataset to generate its feature .pkl file 

Use the ```train.pkl``` and ```test.pkl``` to train the model and save the custom model to ```model.model```. The model and ```deg_model.pkl``` will be saved in ```Model``` folder and ```model_MSE_result.csv``` will be saved in ```Results``` folder. 

```bash
python GH_Train.py train.pkl test.pkl model.model
```
Run ```GH_Predict.py``` but use the custom model instead of the provided pretrain model to predict pIC50 of selected dataset.

