import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rot_flip3d(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 0))
    label = np.rot90(label, k, axes=(1, 0))
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(1, 2))
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(2, 0))
    label = np.rot90(label, k, axes=(2, 0))
    axis = np.random.randint(0, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_rotate3d(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(1, 0), order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=(1, 0), order=0, reshape=False)
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(1, 2), order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=(1, 2), order=0, reshape=False)
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(2, 0), order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=(2, 0), order=0, reshape=False)
    return image, label


class RandomGenerator_patches(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        # take a random 3d patch instead of resizing the whole image
        maxx = x - self.output_size[0]
        maxy = y - self.output_size[1]
        xstart = random.randint(0, maxx)
        ystart = random.randint(0, maxy)
        image = image[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1]]
        label = label[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1]]

        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class RandomCrop3D():
    def __init__(self, img_sz, crop_sz):
        c, h, w, d = img_sz
        assert (h, w, d) > crop_sz
        self.img_sz = tuple((h, w, d))
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return None, None

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class RandomGenerator3d_patches(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y, z = image.shape
        # take a random 3d patch instead of resizing the whole image
        maxx = x - self.output_size[0]
        maxy = y - self.output_size[1]
        maxz = z - self.output_size[2]
        xstart = random.randint(0, maxx)
        ystart = random.randint(0, maxy)
        zstart = random.randint(0, maxz)
        image = image[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]
        label = label[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]

        if random.random() > 0.5:
            image, label = random_rot_flip3d(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate3d(image, label)

        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator3d_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1] or z != self.output_size[2]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z),
                         order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=0)

        if random.random() > 0.5:
            image, label = random_rot_flip3d(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate3d(image, label)

        if torch.is_tensor(image):
            pass
        else:
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


class Reshape3d_patches(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, z = image.shape
        # take a random 3d patch instead of resizing the whole image
        maxx = x - self.output_size[0]
        maxy = y - self.output_size[1]
        maxz = z - self.output_size[2]
        xstart = random.randint(0, maxx)
        ystart = random.randint(0, maxy)
        zstart = random.randint(0, maxz)
        image = image[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]
        label = label[xstart:xstart + self.output_size[0], ystart:ystart + self.output_size[1],
                zstart:zstart + self.output_size[2]]
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Reshape3d_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z),
                         order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Reshape_zoom(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
