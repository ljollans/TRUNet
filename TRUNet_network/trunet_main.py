import argparse
import ml_collections
import os
from glob import glob
import torch
import numpy as np
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from TRUNet.TRUNet_network.model.ViT import VisionTransformer3d as TransUNet3d
from datetime import datetime
from TRUNet_network.trunet_train import trainer
from torchvision import transforms
from TRUNet_network.augmentations import RandomGenerator3d_zoom, Reshape3d_zoom

now = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=6, help='number of class')
parser.add_argument('--batch_size', type=int,
                    default=2, help='training batch size')
parser.add_argument('--root_path', type=str,
                    default='/proj/berzelius-2023-86/new_LiU3D/', help='path with train and val directories')
parser.add_argument('--save_path', type=str,
                    default='./run_' + now.strftime("%m%d%Y_%H%M%S"), help='path to save outputs')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='number of epochs')
parser.add_argument('--crop', type=str,
                    default='/proj/berzelius-2023-86/segmentation_runs/models/crop_train.txt',
                    help='crop coordinate list')
parser.add_argument('--checkpoint', type=str,
                    default='None', help='path to partially trained model')

args_ = parser.parse_args()


def TransUNet_configs(img_size):
    configs_trunet = ml_collections.ConfigDict()

    configs_trunet.resnet = ml_collections.ConfigDict()
    configs_trunet.resnet.num_layers = (3, 4, 9)
    configs_trunet.resnet.width_factor = 1
    configs_trunet.transformer_mlp_dim = 3072
    configs_trunet.transformer_num_heads = 12
    configs_trunet.transformer_num_layers = 12
    configs_trunet.transformer_attention_dropout_rate = 0.0
    configs_trunet.transformer_dropout_rate = 0.1
    configs_trunet.classifier = 'seg'
    configs_trunet.decoder_channels = (256, 128, 64, 16)
    configs_trunet.n_classes = 7
    configs_trunet.n_skip = 3
    configs_trunet.skip_channels = [512, 256, 64, 16]
    configs_trunet.patches = ml_collections.ConfigDict()
    configs_trunet.patches.grid = None

    configs_trunet.hidden_size = 768
    configs_trunet.patches.size = 16

    configs_trunet.patch_size = configs_trunet.patches.size  # (results in 14 by 14 grid of patches for input size 224)

    configs_trunet.patches.grid = (
        int(img_size / configs_trunet.patches.size), int(img_size / configs_trunet.patches.size),
        int(img_size / configs_trunet.patches.size))
    configs_trunet.hybrid = True

    return configs_trunet


class fetch_dataset:
    def __init__(self, base_dir, crop=None, transform=None):
        self.transform = transform
        self.data_dir = base_dir
        sample_list = sorted(glob(os.path.join(base_dir, '*.npz')))
        self.sample_list = sample_list
        self.name = [i.split('/')[-1].split('.npz')[0] for i in sample_list]
        self.pt = [int(i.split('_')[0][2:]) for i in self.name]
        self.crop = crop
        if crop is None or crop == 'None':
            pass
        else:
            with open(crop) as file:
                lines = [line.rstrip() for line in file]
            self.crop_pt = [int(i.split(' ')[0]) for i in lines]
            self.crop_xmin = [float(i.split(' ')[1]) for i in lines]
            self.crop_xmax = [float(i.split(' ')[2]) for i in lines]
            self.crop_ymin = [float(i.split(' ')[3]) for i in lines]
            self.crop_ymax = [float(i.split(' ')[4]) for i in lines]
            self.crop_zmin = [float(i.split(' ')[5]) for i in lines]
            self.crop_zmax = [float(i.split(' ')[6]) for i in lines]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data_path = self.sample_list[idx]
        data = np.load(data_path)
        image, label = data['arr_0'], data['arr_1']

        if self.crop is None or self.crop == 'None':
            pass
        else:
            idx_crop = self.crop_pt.index(self.pt[idx])
            idx_crop = int(idx_crop)
            x, y, z = image.shape
            xmin = max([0, int(self.crop_xmin[idx_crop]) - 20])
            ymin = max([0, int(self.crop_ymin[idx_crop]) - 20])
            zmin = max([0, int(self.crop_zmin[idx_crop]) - 20])
            xmax = min([x, int(self.crop_xmax[idx_crop]) + 20])
            ymax = min([y, int(self.crop_ymax[idx_crop]) + 20])
            zmax = min([z, int(self.crop_zmax[idx_crop]) + 20])
            image = image[xmin:xmax, ymin:ymax, zmin:zmax]
            label = label[xmin:xmax, ymin:ymax, zmin:zmax]

        sample = {'image': image, 'label': label, 'case_name': self.sample_list[idx].split('/')[-1].split('.npz')[0]}
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    args = ml_collections.ConfigDict()
    args.max_epochs = args_.max_epochs
    args.save_path = args_.save_path
    args.root_path = args_.root_path
    args.crop = args_.crop
    args.num_classes = args_.num_classes
    args.batch_size = args_.batch_size
    args.seed = 42
    args.base_lr = 0.01

    # Transforms & Augmentations
    train_transforms = transforms.Compose(
        [RandomGenerator3d_zoom(output_size=(args.img_size, args.img_size, args.img_size))])
    val_transforms = transforms.Compose([Reshape3d_zoom(output_size=[args.img_size, args.img_size, args.img_size])])

    # Define model
    config_net = TransUNet_configs(args.img_size)
    model = TransUNet3d(config_net, img_size=args.img_size, num_classes=args.num_classes, zero_head=False,
                        vis=False)
    config = {'ds_val': fetch_dataset(base_dir=os.path.join(args.root_path, 'val'), transform=val_transforms,
                                      crop=args_.crop),
              'ds_train': fetch_dataset(base_dir=os.path.join(args.root_path, 'train'),
                                        transform=train_transforms, crop=args_.crop),
              'loss_function': DiceCELoss(include_background=False, to_onehot_y=True, softmax=True),
              'metric': DiceMetric(include_background=False, reduction="mean"),
              'optimizer': torch.optim.Adam(model.parameters(), args.base_lr),
              'save_interval': 50}

    if args_.checkpoint == 'None':
        pass
    else:
        print('loading checkpoint ', args_.checkpoint)
        model_state = torch.load(args_.checkpoint, map_location='cpu')
        model.load_state_dict(model_state)

    trainer(args, config, model, args.save_path)
