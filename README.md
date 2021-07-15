# Multibind-RTKs
## Usage
### Prerequisites
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
Generate 16 Fingerprints from ```dataset.csv``` concat and save it to ```dataset_FP.csv```

```bash
python GH_Fingerprint_Generation.py dataset.csv dataset_FP.csv
```

The 16 csv file of fingerprint will collect in ```Fingerprints``` separately.
