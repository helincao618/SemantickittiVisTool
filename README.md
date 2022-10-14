# Visualization & Evaluation SemanticKITTI
### Data split
train['00','01','02','03','04','05','06','07','09','10']

val['08']

test['11','12','13','14','15','16','17','18','19','20','21']

### File organization
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
# SemantickittiVisTool
