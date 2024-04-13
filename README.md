## Installation
This project is an extension of Segformer, please refer to https://github.com/NVlabs/SegFormer/tree/master for backbone and copy files to corresponding folders.
Env setup example (works on 4070 laptop)
```
conda create -n env_name python=3.7 -y
conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
cd SegFormer && pip install -e . --user
pip install iPython
pip install attrs
pip install timm==0.3.2
```
change this line in anaconda3/envs/env_name/lib/python3.7/site-packages/timm/models/layers/helpers.py 
original : #from torch._six import container_abcs
change to : import collections.abc as container_abcs

## Dataset
The RUGD and RELLIS-3D need relabling with the python script in dataset folder. In related python scripts the paths are absolute paths so please modify them.
The structures are as follows
```
/home/username
├── Dataset
│   ├── RELLIS-3D
│   │   │── test.txt
│   │   │── train.txt
│   │   │── val.txt
│   │   │── annotation
│   │   │   ├── 00000 & 00001 & 00002 & 00003 & 00004 
│   │   │── image
│   │   │   ├── 00000 & 00001 & 00002 & 00003 & 00004 
│   ├── RUGD
│   │   │── test_ours.txt
│   │   │── train_ours.txt
│   │   ├── creek & park-1/2/8 & trail-(1 & 3-7 & 9-15) & village
│   │   │── RUGD_annotations
│   │   │   ├── creek & park-1/2/8 & trail-(1 & 3-7 & 9-15) & village
```

## Train, Evalution, Visualization
Please refer to segformer.
