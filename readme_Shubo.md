## Setup
```
conda create --name HuPR python=3.7.10
conda activate HuPR
python setup.py
```
Extract the dataset in the 'preprocessing/raw_data/iwr1843'  (move in HuPR/)

Note: 
- The required packages are in `requirements.txt`
- Don't update the COCO Python API into the repository. COCO evaluation code has 17 keypoints, but the ground truth and predicted keypoints have 14 points in HuPR.

## Preprocessing
```
cd preprocessing
nohup python process_iwr1843.py
```
Note:
- There are 276 sequences (named as `single_xx`), with each having 600 frames. The preprocessed data is stored in `data/HuPR/`
- The preprocessing is very slow and gets killed sometimes
- The index of data to be preprocessed can be change in `for idxName in range()`
- The preprocessing cannot be excecuted multiple copies in parallel (?)

## Training and Evaluation
```
python main.py --config mscsa_prgcn.yaml --dir mscsa_prgcn # -sr 10
python main.py --config mscsa_prgcn.yaml --dir mscsa_prgcn --eval
```
Note:
- The train/evaluat/test data indexes are in `config/mscsa_prgcn.yml`


## Problems:
- To train the entire network, we need all the data and preprocess them
- Preprocessing is very slow and sometimes gets killed randomly
- The preprocessing cannot be excecuted multiple copies in parallel (?)
- The training batch size too large and results in out of GPU memory. I changed to batch size 5




