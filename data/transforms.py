import random
import torch
import torchvision
import torchvision.transforms.functional as TF


class ToTensor(object):
    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    @staticmethod
    def encode_segmentation_map(mask):
        labels_map = torch.zeros(mask.shape)
        labels_map[mask > 0] = 1

        return labels_map.to(dtype=torch.int64)

    def __call__(self, sample):
        image, mask = sample
        return self.transform(image), self.encode_segmentation_map(self.transform(mask))


class Resize(object):
    def __init__(self, size):
        self.resize = torchvision.transforms.Resize(size,
                                                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        image, mask = sample
        return self.resize(image), self.resize(mask)


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.flip = lambda image: TF.hflip(image) if random.random() < p else image

    def __call__(self, sample):
        image, mask = sample
        return self.flip(image), self.flip(mask)


class RandomRotation(object):
    def __init__(self, degrees):
        angle = torchvision.transforms.RandomRotation.get_params((-degrees, degrees))
        self.rotate = lambda image: TF.rotate(image, angle)

    def __call__(self, sample):
        image, mask = sample
        return self.rotate(image), self.rotate(mask)


class RandomScale(object):
    def __init__(self, scale):
        self.scale = scale

    def scale(self, image):
        ret = torchvision.transforms.RandomAffine.get_params((0, 0), None, self.scale, None, image.size)
        return TF.affine(image, *ret, resample=False, fillcolor=0)

    def __call__(self, sample):
        image, mask = sample
        return self.scale(image), self.scale(mask)


class BrightContrastJitter(object):
    def __init__(self, brightness=0, contrast=0):
        self.transform = torchvision.transforms.ColorJitter(brightness, contrast, 0, 0)

    def __call__(self, sample):
        image, mask = sample
        return self.transform(image), mask
