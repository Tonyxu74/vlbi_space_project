import glob
from torch.utils import data
import torchvision.transforms as transforms
from itertools import chain
from PIL import Image
from myargs import args
import torch

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
    def __init__(self, impath, eval, datatype):
        'Initialization'

        self.eval = eval
        self.datatype = datatype
        self.std = 0

        # add images to dataset
        datalist = []
        imgpaths = glob.glob('{}/*.jpg'.format(impath))
        for impath in imgpaths:
            datalist.append({'image': impath})
        self.datalist = datalist
        self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

        # define transforms
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.imageDims, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        self.gauss_blur = transforms.GaussianBlur(kernel_size=15, sigma=(5.0, 7.0))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'

        # load image
        orig_image = Image.open(self.datalist[index]).convert('L')
        orig_image = self.transforms(orig_image)

        # generate UV plane and dirty beam
        uv_coverage = uv_generate(args.imageDims[0])
        dirty_beam = self.gauss_blur(torch.abs(ifft_centered(uv_coverage)))

        # get dirty image
        fft_orig = fft_centered(orig_image)
        dirty_img = self.gauss_blur(torch.abs(ifft_centered(fft_orig * uv_coverage)))

        # normalize image and beam between 0 and 1, and concatenate
        dirty_img = (dirty_img - torch.min(dirty_img)) / (torch.max(dirty_img) - torch.min(dirty_img))
        dirty_beam = (dirty_beam - torch.min(dirty_beam)) / (torch.max(dirty_beam) - torch.min(dirty_beam))

        dirty_input = torch.cat((dirty_img, dirty_beam), dim=0)

        # return image and label
        return dirty_input, orig_image
