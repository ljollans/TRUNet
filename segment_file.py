################################################################################################
# Automatic cardiac CT volume segmentation
# Author: Lee Jollans, 2023
# work described in the manuscript "faster 3D volume segmentation with vision transformers"
################################################################################################


# INPUT: DICOM directory (or .nii file)

# OPTIONS: - TRUNet (default) or residual UNet segmentation:
#                                  --unet flag or no flag
#          - Resizing to original image size (default) or resizing:
#                                  --size XX,YY,ZZ or no flag
#          - Cropping to bounding box or not (default). This is done after the segmentation is calculated
#                                  --crop
#          - Resaving input DICOM as .nii as well or not (default)
#                                  --saveimg or no flag
#          - Retaining only clusters above a certain cluster threshold (default) or retaining all clusters
#                                  --clusterthresh 0 (to keep all clusters) or e.g. --clusterthresh 1000 (to keep clusters with >1000 voxels)
#          - calculation to be run on cpu (default) or gpu:
#                                  --cuda / --gpu or --cpu

# OUTPUT: initially .nii, also .stl

# additional functionality to add later:
#          - change coordinate system
#          - convert multiple volumes at once
#          - uncertainty estimation
#          - dice score when a ground truth is available

# use note: unsure why, but if another instance of python is open sometimes the inference process gets killed!

# usage example:
# python segment_file.py /Users/emijo30/OneDrive/data/CT_from_mscproj/nifti/pt16/ct_pt16dt8.nii.gz

import os
import sys
import nibabel as nib
from glob import glob
import pydicom
import numpy as np
from TRUNet_network.configs import trunetConfigs
from TRUNet_network.model.ViT import VisionTransformer3d as TRUNet
import torch
from monai.networks.nets import UNet as MONAIUNet
from scipy.ndimage import zoom
import time
import argparse
from scipy.ndimage import label

numClasses = 7
device = 'cpu'
network = 'trunet'
resize = False
imageSize = None
savePath = None
saveImg = False
crop = False
threshold = 1000000


def parse_size(size_arg):
    try:
        size_list = list(map(int, size_arg.split(',')))
        if len(size_list) != 3:
            raise ValueError
        return size_list
    except ValueError:
        raise argparse.ArgumentTypeError("Size must be three comma-separated integers")


################################################################################################
# load the dataset

def transform_to_hu(dicom_image):
    """
    Function for transforming the dicom pixel values to Hounsfield Units.
    Inspiration Source: https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
    :param dicom_image: dicom that has been opened with pydicom.dcmread()
    :return: np.array with converted values.
    note by LJ: taken from MSc project code
    """
    pixel_image = dicom_image.pixel_array
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope
    hu_image = pixel_image * slope + intercept
    return hu_image


def dicom_to_numpy(dicom_path):
    # note by LJ: taken from MSc project code
    slices = [pydicom.dcmread(os.path.join(dicom_path, s)) for s in os.listdir(dicom_path)]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=True)
    ct_slices = [transform_to_hu(s).astype(np.int16) for s in slices]
    data = np.squeeze(ct_slices).T
    return data


def load_ct_image(imagePath):
    # the given path is a .nii file
    if os.path.isfile(imagePath):
        img = nib.load(imagePath).get_fdata()
        savePath = os.path.dirname(imagePath)
    else:
        # the given path is a folder
        savePath = imagePath
        listNii = glob(os.path.join(imagePath, '*.nii.gz'))

        # the folder does not contain .nii files
        if len(listNii) == 0:  # read dicom files
            img = dicom_to_numpy(imagePath)
            print('loaded dicom series from ' + imagePath, end="...")

        # the folder contains only one .nii file
        elif len(listNii) == 1:  # only one file, read it
            img = nib.load(listNii[0]).get_fdata()
            print('loaded ' + listNii[0], end="...")

        # the folder contains multiple .nii files
        else:
            filePrefix = [i.split('/')[-1][:2] for i in listNii]
            if any([i == 'ct' for i in filePrefix]):
                img = nib.load(listNii[filePrefix.index('ct')]).get_fdata()
                print('loaded ' + listNii[filePrefix.index('ct')], end="...")
            else:
                print('could not identify the correct nii file')

    print(f'image shape: {img.shape[0]} * {img.shape[1]} * {img.shape[2]}')

    return img, savePath


################################################################################################
# load the model

def get_model(network, modelDirectory, device=device):
    if network == 'trunet':

        config_net = trunetConfigs('3d', 224)
        model = TRUNet(config_net, img_size=224, num_classes=numClasses)

    elif network == 'unet':
        model = MONAIUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=numClasses,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    modelPath = os.path.join(modelDirectory, network + '_model.pth')
    print('loading model from ', modelPath)
    model_state = torch.load(modelPath, map_location=torch.device(device))
    model.load_state_dict(model_state)

    return model


################################################################################################
# run segmentation

def test_single_volume(image, net, device=device, newImageShape=imageSize):
    # input to the model should be of shape 1,1,224,224
    patch_size = [224, 224, 224]
    with torch.no_grad():
        x, y, z = image.shape[0], image.shape[1], image.shape[2]
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y, patch_size[2] / z), order=3)  # previous using 0
        if device == 'cuda':
            input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        net.eval()
        with torch.no_grad():
            outputs = net(input.float())
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            if newImageShape is not None:
                x, y, z = newImageShape[0], newImageShape[1], newImageShape[2]
            pred = zoom(out, (x / patch_size[0], y / patch_size[1], z / patch_size[2]), order=0)
            outputs = zoom(outputs.cpu().detach().numpy(),
                           (1 / 1, numClasses / numClasses, x / patch_size[0], y / patch_size[1], z / patch_size[2]),
                           order=0)
    return pred, outputs


def resizeImage(image, newImageShape=imageSize):
    x, y, z = newImageShape[0], newImageShape[1], newImageShape[2]
    image = zoom(image, (x / image.shape[0], y / image.shape[1], z / image.shape[2]), order=0)
    return image


def get_certainty(pred, outputs):
    certMat = np.zeros(pred.shape)
    for roi in range(numClasses - 1):
        certs = outputs[0, roi]
        certMat[pred == roi] = certs[pred == roi]
    return certMat


################################################################################################
# remove clusters below threshold

def labels(seg, roi):
    seg1 = np.where(seg == roi, 1, 0)
    labelled, num_labels = label(seg1)
    size_label = [len(np.where(labelled == n)[0]) for n in range(num_labels + 1)]
    return labelled, size_label


def keep_only_largest(seg, threshold):
    newseg = np.zeros(seg.shape)
    for roi in range(numClasses - 1):
        labelled, size_label = labels(seg, roi)
        toKeep = []
        toKeepSize = []
        for clus in range(len(size_label)):
            if clus > 0:
                if size_label[clus] > threshold:
                    toKeep.append(clus)
                    toKeepSize.append(size_label[clus])
        # print(f'ROI {roi}: retaining {len(toKeep)} clusters {toKeepSize}  voxels large.')
        if len(toKeep) == 0:
            print('No clusters remain for ROI', roi, end="...")
        for clus in toKeep:
            newseg[np.where(labelled == clus)] = roi

    return newseg

def cropToBbox(img, prediction):
    a = np.where(prediction > 0)
    x, y, z = prediction.shape
    xmin = max([0, int(np.nanmin(a[0])) - 20])
    ymin = max([0, int(np.nanmin(a[1])) - 20])
    zmin = max([0, int(np.nanmin(a[2])) - 20])
    xmax = min([x, int(np.nanmax(a[0])) + 20])
    ymax = min([y, int(np.nanmax(a[1])) + 20])
    zmax = min([z, int(np.nanmax(a[2])) + 20])
    img = img[xmin:xmax, ymin:ymax, zmin:zmax]
    prediction = prediction[xmin:xmax, ymin:ymax, zmin:zmax]
    return img, prediction


################################################################################################
# save the output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script description")

    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("--unet", action="store_true", help="Use unet network")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument("--size", type=parse_size, help="Resize image size (comma-separated integers)")
    parser.add_argument("--saveimg", action="store_true", help="Resave the original volume as .nii")
    parser.add_argument("--crop", action="store_true",
                        help="Crop the segmentation to the ROI and re-save cropped volume")
    parser.add_argument("--clusterthresh", type=int, help="Minimum size of clusters for them to be retained")
    parser.add_argument("--prefix", type=str, help="File name prefix")

    args = parser.parse_args()

    imagePath = args.image_path
    print('loading image from ', imagePath)

    if args.unet:
        network = 'unet'
        print('Inference with the residual UNet model.')
    else:
        print('Inference with the TRUNet model.')

    if args.gpu or args.cuda:
        device = 'cuda'
    print('Running on', device)

    if args.size is not None:
        resize = True
        imageSize = args.size
        print('Images will be resized to', imageSize)

    if args.saveimg:
        saveImg = True
        print('Image will be resaved as .nii.gz file')


    ################################################################################################
    # load the dataset
    img, savePath = load_ct_image(imagePath)

    ################################################################################################
    # load the model
    script_directory = os.path.dirname(os.path.abspath(__file__))
    modelDirectory = os.path.join(script_directory, 'models')
    model = get_model(network, modelDirectory)

    ################################################################################################
    # run segmentation
    print('running model', end="...")
    start_time = time.time()
    if resize:
        prediction, outputs = test_single_volume(img, model, device, newImageShape=imageSize)
        img = resizeImage(img, newImageShape=imageSize)
    else:
        prediction, outputs = test_single_volume(img, model, device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{elapsed_time:.6f} seconds")

    ################################################################################################
    # remove clusters below threshold

    if args.clusterthresh is not None:
        threshold = args.clusterthresh
    else:
        if network == 'trunet':
            threshold = 0
        else:
            totalVoxels = img.shape[0] * img.shape[1] * img.shape[2]
            threshold = totalVoxels * 0.0005

    if threshold > 0:
        print('removing small clusters', end="...")
        start_time = time.time()
        prediction = keep_only_largest(prediction, threshold)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{elapsed_time:.6f} seconds")

    ################################################################################################
    # crop segmentation and image to ROI

    if args.crop:
        img, prediction = cropToBbox(img, prediction)
        saveImg = True

    ################################################################################################
    # show some plots
    # certainty = get_certainty(prediction, outputs)
    # # plt.imshow(prediction[:,:,300]); plt.show()
    # fig = plt.figure(figsize=[10, 10])
    # for d in range(7):
    #     plt.subplot(3, 3, d + 1)
    #     plt.imshow(outputs[0, d, 300, :, :], clim=[0, 10]);
    #     plt.colorbar()
    # plt.subplot(3, 3, 8)
    # plt.imshow(prediction[300, :, :]);
    # plt.colorbar()
    # plt.subplot(3, 3, 9)
    # plt.imshow(np.nanmin(certainty,0));
    # plt.colorbar()
    # plt.show()

    ################################################################################################
    # save the output

    if args.prefix is not None:
        savePathFull = os.path.join(savePath, args.prefix + '_' + network + '_segmentation.nii.gz')
    else:
        savePathFull = os.path.join(savePath, network + '_segmentation.nii.gz')
    print('saving segmentation to', savePathFull)
    seg = nib.Nifti1Image(prediction.astype(np.float32), np.eye(4))
    nib.save(seg, savePathFull)

    if saveImg:
        if args.prefix is not None:
            savePathFull = os.path.join(savePath, args.prefix + '_' + network + '_volume.nii.gz')
        else:
            savePathFull = os.path.join(savePath, 'volume.nii.gz')
        print('saving volume to', savePathFull)
        img = nib.Nifti1Image(img.astype(np.float32), np.eye(4))
        nib.save(img, savePathFull)