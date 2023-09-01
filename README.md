# TRUNet
Semantic segmentation for 3D volume images using a modified ResNet50 v2 block and a Vision Transformer Block in a U-Net framework.

This is an adaptation of [TransUNet](https://arxiv.org/abs/2102.04306) for 3D inputs. Instead of the CNN encoder used in TransUNet the "Hybrid" approach including a modified RedNet50 block proposed by the authors is used.

![TRUNet Architecture](https://github.com/ljollans/TRUNet/blob/main/TRUNet_network/TRUNet_architecture.png)


Installation:
- Clone repository: ``git clone https://github.com/ljollans/TRUNet.git``
- If you want to create a new virtual environment:
    - ``python3 -m venv ./venv``
    - ``source ./venv/bin/activate``
- Install requirements: ``pip install -r requirements.txt``
  
[22nd August 2023]
currently the cardiac segmentation model trained using TRUNet is not available because of its large size
the manuscript to accompany this network is in preparation
a recent [similar publication](https://www.nature.com/articles/s41598-023-40841-y) also reported on a cardiac segmentation task using a 3D implementation of TransUNet
