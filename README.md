# TRUNet
Pre-print: [here
](https://arxiv.org/abs/2310.09099)

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

Related work:

[Li, Dapeng, et al. "A task-unified network with transformer and spatial–temporal convolution for left ventricular quantification." Scientific Reports 13.1 (2023): 13529.](https://www.nature.com/articles/s41598-023-40841-y) 

[Chen, Jieneng, et al. "3d transunet: Advancing medical image segmentation through vision transformers." arXiv preprint arXiv:2310.07781 (2023).](https://arxiv.org/abs/2310.07781)

[Yang, Siwei, et al. "3D-TransUNet for Brain Metastases Segmentation in the BraTS2023 Challenge." arXiv preprint arXiv:2403.15735 (2024).](https://arxiv.org/abs/2403.15735)
