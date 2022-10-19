# Visualization & Evaluation SemanticKITTI
## Data split
train['00','01','02','03','04','05','06','07','09','10']

val['08']

test['11','12','13','14','15','16','17','18','19','20','21']

## File organization
```angular2
dataset
├── sequences
│  ├── 00
│  │  ├── labels
│  │  ├── velodyne
│  │  ├── voxels
│  │  ├── [OTHER FILES OR FOLDERS]
│  ├── 01
│  ├── ... ...

```
## Visualization
### Visualize the training data
```
python train_visualization.py --iffusion --startframe --endframe --seqpath
```
### Visualize/Save the test/val data
```
python train_visualization.py --iffusion --startframe --endframe --seqpath
```
### Visualize/Save the input data
```
python input_save.py --iffuse True --ifsave True
```
### Read the visualize the data from the file
```
python testvis_fromfile.py --seq '08' --predall True --predstatic True --input True --pose True
```
### Evaluation Semantic Scene Completion
```
python evaluation_completion.py --dataset PATH-TO-ORIGINAL-DATASET --predictions PATH-TO-PREDICTION --split valid --datacfg semantic-kitti.yaml
```

# JS3C-Net
## Installation
* Check if you have installed nvcc & cmake & cudnn correctly
* Clone the JS3C-Net
```
git clone https://github.com/yanx27/JS3C-Net.git --recursive
```
* Set the virtual environment using conda
```
conda create -n js3c python=3.7
conda activate js3c
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
cd JS3C-Net/lib
pip install cython
sh complile.sh in /lib
sudo apt-get install libboost-all-dev
```

* Install the spconv
  
Add the fllowing code in the second line of CMakelist
```
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.1/bin/nvcc") 
```
Compile and install the spconv
```
cd spconv/
python setup.py bdist_wheel
cd dist/
pip install .whl
conda install pyyaml==5.4.1 tqdm scipy
```

## Train & Test
### Training
```
python train.py --log_dir JS3C-Net-kitti --gpu 0 --config ./JS3C-Net/opt/JS3C_default_kitti.yaml"
```
### Validation
```
python test_kitti_ssc.py --log_dir ./JS3C-Net/log/JS3C-Net-kitti --gpu 0 --dataset valid
```
### Test
```
python test_kitti_ssc.py --log_dir ./JS3C-Net/log/JS3C-Net-kitti --gpu 0 --dataset test
```

