import glob
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch
from myargs import args
import math
from utils.UV_plane_generator import uv_generate

AMP_MEAN = (0.500,)
AMP_STD = (0.500,)

PHASE_MEAN = (0.,)
PHASE_STD = (2*math.pi,)

eps = 1e-8


def findFile(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval, datatype):
        'Initialization'

        self.eval = eval
        self.datatype = datatype
        self.std = 0

        # add images in different folders to the datalist
        datalist = []
        folders = glob.glob('{}/*/'.format(impath))

        for imfolder in folders:
            amp_image_list = sorted(findFile(imfolder, 'amp_img.npy'))
            phase_image_list = sorted(findFile(imfolder, 'phase_img.npy'))
            amp_gt_list = sorted(findFile(imfolder, 'amp_gt.npy'))
            phase_gt_list = sorted(findFile(imfolder, 'phase_gt.npy'))
            datalist.append([{
                'amp_image': amp_img,
                'phase_image': phase_img,
                'amp_label': amp_gt,
                'phase_label': phase_gt
            } for amp_img, phase_img, amp_gt, phase_gt in zip(amp_image_list, phase_image_list, amp_gt_list, phase_gt_list)
            ])
        self.datalist = [item for sublist in datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label given datatype
        if self.datatype == 'amp':
            image = np.load(self.datalist[index]['amp_image'])
            label = np.load(self.datalist[index]['amp_label'])

        elif self.datatype == 'phase':
            image = np.load(self.datalist[index]['phase_image'])
            label = np.load(self.datalist[index]['phase_label'])

        elif self.datatype == 'comb':
            amp_image = np.load(self.datalist[index]['amp_image'])
            phase_image = np.load(self.datalist[index]['phase_image'])
            amp_label = np.load(self.datalist[index]['amp_label'])
            phase_label = np.load(self.datalist[index]['phase_label'])

            amp_image, amp_label = normalizepatch(amp_image, amp_label, self.eval, self.std, 'amp')
            phase_image, phase_label = normalizepatch(phase_image, phase_label, self.eval, self.std, 'phase')

            return amp_image, phase_image, amp_label, phase_label

        else:
            raise Exception("Improper datatype called, only 'amp' or 'phase' or 'comb' permitted, {} given".format(self.datatype))

        # augmentations on image
        image, label = normalizepatch(image, label, self.eval, self.std, datatype=self.datatype)

        if label.shape[-1] < 256:
            upsample = torch.nn.Upsample(size=(256, 256))
            label = upsample(label.unsqueeze(0))[0]

        return image, label


def GenerateIterator(args, impath, eval=False, shuffle=True, datatype='amp'):
    params = {
        'batch_size': args.batchSize,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=eval, datatype=datatype), **params)


class Dataset_val(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval, datatype):
        'Initialization'

        self.eval = eval
        self.datatype = datatype
        self.std = 0

        # add images in different folders to the datalist
        datalist = []
        folders = glob.glob('{}/*/'.format(impath))

        for imfolder in folders:
            amp_image_list = sorted(findFile(imfolder, 'amp_img.npy'))
            phase_image_list = sorted(findFile(imfolder, 'phase_img.npy'))
            amp_gt_list = sorted(findFile(imfolder, 'amp_gt.npy'))
            phase_gt_list = sorted(findFile(imfolder, 'phase_gt.npy'))
            datalist.append([{
                'amp_image': amp_img,
                'phase_image': phase_img,
                'amp_label': amp_gt,
                'phase_label': phase_gt
            } for amp_img, phase_img, amp_gt, phase_gt in zip(amp_image_list, phase_image_list, amp_gt_list, phase_gt_list)
            ])
        self.datalist = [item for sublist in datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label given datatype
        if self.datatype == 'amp':
            image = np.load(self.datalist[index]['amp_image'])
            label = np.load(self.datalist[index]['amp_label'])

        elif self.datatype == 'phase':
            image = np.load(self.datalist[index]['phase_image'])
            label = np.load(self.datalist[index]['phase_label'])

        elif self.datatype == 'comb':
            amp_image = np.load(self.datalist[index]['amp_image'])
            phase_image = np.load(self.datalist[index]['phase_image'])
            amp_label = np.load(self.datalist[index]['amp_label'])
            phase_label = np.load(self.datalist[index]['phase_label'])

            amp_image, amp_label = normalizepatch(amp_image, amp_label, self.eval, self.std, 'amp')
            phase_image, phase_label = normalizepatch(phase_image, phase_label, self.eval, self.std, 'phase')

            return amp_image, phase_image, amp_label, phase_label

        else:
            raise Exception("Improper datatype called, only 'amp' or 'phase' or 'comb' permitted, {} given".format(self.datatype))

        # augmentations on image
        image, label = normalizepatch(image, label, self.eval, self.std, datatype=self.datatype)

        if label.shape[-1] < 256:
            upsample = torch.nn.Upsample(size=(256, 256))
            label = upsample(label.unsqueeze(0))[0]

        return image, label


def GenerateIterator_val(args, impath, shuffle=True, datatype='amp'):
    params = {
        'batch_size': args.batchSize,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset_val(impath, eval=True, datatype=datatype), **params)


class Dataset_train(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, impath, eval, datatype):
        'Initialization'

        self.eval = eval
        self.datatype = datatype
        self.std = 0

        # add images in different folders to the datalist
        datalist = []
        folders = glob.glob('{}/*/'.format(impath))

        for imfolder in folders:
            amp_image_list = sorted(findFile(imfolder, 'amp_img.npy'))
            phase_image_list = sorted(findFile(imfolder, 'phase_img.npy'))
            datalist.append([{
                'amp_image': amp_img,
                'phase_image': phase_img,
            } for amp_img, phase_img in zip(amp_image_list, phase_image_list)
            ])
        self.datalist = [item for sublist in datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 1 for i in self.datalist]))

        self.simulated_uv = uv_generate(uvnum=10, output_size=256, rotation=120)

    def generate_uv(self):
        simulated_uv = uv_generate(uvnum=10, output_size=256, rotation=120)
        self.simulated_uv = simulated_uv

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.datalist)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label given datatype
        if self.datatype == 'amp':
            label = np.load(self.datalist[index]['amp_image'])
            rand_uv = self.simulated_uv[np.random.randint(0, len(self.simulated_uv))]
            image = label * rand_uv

        elif self.datatype == 'phase':
            label = np.load(self.datalist[index]['phase_image'])
            rand_uv = self.simulated_uv[np.random.randint(0, len(self.simulated_uv))]
            image = label * rand_uv

        elif self.datatype == 'comb':
            amp_label = np.load(self.datalist[index]['amp_image'])
            phase_label = np.load(self.datalist[index]['phase_image'])

            rand_uv = self.simulated_uv[np.random.randint(0, len(self.simulated_uv))]

            amp_image = amp_label * rand_uv
            phase_image = phase_label * rand_uv

            amp_image, amp_label = normalizepatch(amp_image, amp_label, self.eval, self.std, 'amp')
            phase_image, phase_label = normalizepatch(phase_image, phase_label, self.eval, self.std, 'phase')

            return amp_image, phase_image, amp_label, phase_label

        else:
            raise Exception("Improper datatype called, only 'amp' or 'phase' or 'comb' permitted, {} given".format(self.datatype))

        # augmentations on image
        image, label = normalizepatch(image, label, self.eval, self.std, datatype=self.datatype)

        if label.shape[-1] < 256:
            upsample = torch.nn.Upsample(size=(256, 256))
            label = upsample(label.unsqueeze(0))[0]

        return image, label


def GenerateIterator_train(args, impath, eval=False, shuffle=True, datatype='amp'):
    params = {
        'batch_size': args.batchSize,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset_train(impath, eval=eval, datatype=datatype), **params)


def normalizepatch(p, gt, eval, std, datatype):

    if not eval:
        rot_num = random.choice([0, 1, 2, 3])
        p = np.rot90(p, rot_num)
        gt = np.rot90(gt, rot_num)

        noise = np.random.normal(
            0, std, args.imageDims)
        p += noise

    p = np.ascontiguousarray(p)
    gt = np.ascontiguousarray(gt)
    p = torch.from_numpy(p).unsqueeze(0)
    gt = torch.from_numpy(gt).unsqueeze(0)

    if datatype == 'amp':
        return transforms.Normalize(mean=AMP_MEAN, std=AMP_STD)(p.float()), gt.float()

    elif datatype == 'phase':
        return transforms.Normalize(mean=PHASE_MEAN, std=PHASE_STD)(p.float()), \
            transforms.Normalize(mean=PHASE_MEAN, std=PHASE_STD)(gt.float())
