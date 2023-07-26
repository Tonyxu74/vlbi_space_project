import glob
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
from myargs import args
import torch
import numpy as np
from torchvision.transforms import functional as F

from utils.UV_plane_generator import uv_generate


def fft_centered(image, norm='ortho'):
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    fft_image = torch.fft.fft2(image, dim=(-2, -1), norm=norm)
    fft_image = torch.fft.fftshift(fft_image, dim=(-2, -1))
    return fft_image


def ifft_centered(image, norm='ortho'):
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    fft_image = torch.fft.ifft2(image, dim=(-2, -1), norm=norm)
    fft_image = torch.fft.fftshift(fft_image, dim=(-2, -1))
    return fft_image


class TrainImageDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath):
        'Initialization'

        # max radius range allows generally low freq or higher freq coverage
        # sched upper range to be higher during training --> start off with only low freq, then add higher freq
        self.maxrad_range = list(args.maxradWarmup)

        # add images to dataset
        datalist = []
        imgpaths = glob.glob('{}/*/*.jpg'.format(impath))
        for impath in imgpaths:
            datalist.append({'image': impath})
        self.datalist = datalist

        # define transforms
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.imageDims, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        # self.gauss_blur = transforms.GaussianBlur(kernel_size=15)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'

        # load image
        orig_image = Image.open(self.datalist[index]['image']).convert('L')
        orig_image = self.transforms(orig_image)

        # get random radius, random gaussian blur sigma between 2 and 7
        max_rad = np.random.random() * (self.maxrad_range[1] - self.maxrad_range[0]) + self.maxrad_range[0]
        blur_sigma = np.random.random() * 5.0 + 2.0

        # generate UV plane and dirty beam
        uv_coverage = torch.tensor(uv_generate(args.imageDims[0], radius=(0.0, max_rad))).unsqueeze(0)
        dirty_beam = F.gaussian_blur(
            torch.abs(ifft_centered(uv_coverage)), [15, 15], [blur_sigma, blur_sigma])

        # get dirty image
        fft_orig = fft_centered(orig_image)
        dirty_img = F.gaussian_blur(
            torch.abs(ifft_centered(fft_orig * uv_coverage)), [15, 15], [blur_sigma, blur_sigma])

        # normalize image and beam between 0 and 1, and concatenate
        if dirty_img.max() == dirty_img.min():
            dirty_img = torch.ones_like(dirty_img) * 0.5
        else:
            dirty_img = (dirty_img - torch.min(dirty_img)) / (torch.max(dirty_img) - torch.min(dirty_img))
        if dirty_beam.max() == dirty_beam.min():
            dirty_beam = torch.ones_like(dirty_beam) * 0.5
        else:
            dirty_beam = (dirty_beam - torch.min(dirty_beam)) / (torch.max(dirty_beam) - torch.min(dirty_beam))

        dirty_input = torch.cat((dirty_img, dirty_beam), dim=0)

        # return image and label
        return dirty_input.float(), orig_image.float()


class ValidImageDataset(data.Dataset):
    def __init__(self, impath):
        'Initialization'
        # add images to dataset
        datalist = []
        imgpaths = glob.glob('{}/*/*_dirty_img.npy'.format(impath))
        for impath in imgpaths:
            gt_path = impath.replace('_dirty_img', '_target')
            beam_path = impath.replace('_dirty_img', '_dirty_beam')
            datalist.append({'dirty_image': impath, 'label': gt_path, 'dirty_beam': beam_path})
        self.datalist = datalist

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'
        # load dirty image, dirty beam and target
        dirty_img = torch.tensor(np.load(self.datalist[index]['dirty_image']))
        dirty_beam = torch.tensor(np.load(self.datalist[index]['dirty_beam']))
        target = torch.tensor(np.load(self.datalist[index]['label']))

        # normalize image, beam and target between 0 and 1, and stack
        dirty_img = (dirty_img - torch.min(dirty_img)) / (torch.max(dirty_img) - torch.min(dirty_img))
        dirty_beam = (dirty_beam - torch.min(dirty_beam)) / (torch.max(dirty_beam) - torch.min(dirty_beam))
        target = (target - torch.min(target)) / (torch.max(target) - torch.min(target))

        # target swap left right for some reason
        target = torch.flip(target, dims=[-1])

        dirty_input = torch.stack((dirty_img, dirty_beam), dim=0)

        # return image and label
        return dirty_input.float(), target.unsqueeze(0).float()


def GenerateTrainImageIterator(args, impath, shuffle=True):
    params = {
        'batch_size': args.batchSize,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': True,
        'drop_last': False,
    }

    return data.DataLoader(TrainImageDataset(impath), **params)


def GenerateValidImageIterator(args, impath):
    params = {
        'batch_size': args.batchSize,
        'shuffle': False,
        'num_workers': args.workers,
        'pin_memory': True,
        'drop_last': False,
    }

    return data.DataLoader(ValidImageDataset(impath), **params)
