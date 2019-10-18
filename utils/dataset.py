import glob
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch
from myargs import args
from utils.UV_plane_generator import uv_generate

TRAIN_AMP_MEAN = (16.9907,)
TRAIN_AMP_STD = (157.3094,)

TRAIN_PHASE_MEAN = (0.,)
TRAIN_PHASE_STD = (1.8141,)

VAL_AMP_MEAN = (0.00221,)
VAL_AMP_STD = (0.03018,)

VAL_PHASE_MEAN = (0.,)
VAL_PHASE_STD = (1.8104,)

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

            # if amp_label.shape[-1] < 256:
            #     upsample = torch.nn.Upsample(size=(256, 256))
            #     amp_label = upsample(amp_label.unsqueeze(0))[0]
            #
            # if phase_label.shape[-1] < 256:
            #     upsample = torch.nn.Upsample(size=(256, 256))
            #     phase_label = upsample(phase_label.unsqueeze(0))[0]

            return amp_image, phase_image, amp_label, phase_label

        else:
            raise Exception("Improper datatype called, only 'amp' or 'phase' or 'comb' permitted, {} given".format(self.datatype))

        # augmentations on image
        image, label = normalizepatch(image, label, self.eval, self.std, datatype=self.datatype)

        # if label.shape[-1] < 256:
        #     upsample = torch.nn.Upsample(size=(256, 256))
        #     label = upsample(label.unsqueeze(0))[0]

        # nn.functional.interpolate
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


def GenerateIterator_val(args, impath, shuffle=True, datatype='amp'):
    params = {
        'batch_size': args.batchSize,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset(impath, eval=True, datatype=datatype), **params)


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

        self.simulated_uv = uv_generate(uvnum=30, output_size=args.imageDims[0], rotation=120)

    def generate_uv(self, tele_num=10):
        simulated_uv = uv_generate(uvnum=10, telescope_num=tele_num, output_size=args.imageDims[0], rotation=120)
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

        # if label.shape[-1] < 256:
        #     upsample = torch.nn.Upsample(size=(256, 256))
        #     label = upsample(label.unsqueeze(0))[0]

        return image, label


def GenerateIterator_train(args, impath, shuffle=True, datatype='amp'):
    params = {
        'batch_size': args.batchSize,
        'shuffle': shuffle,
        'num_workers': args.workers,
        'pin_memory': False,
        'drop_last': False,
    }

    return data.DataLoader(Dataset_train(impath, eval=False, datatype=datatype), **params)


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

    if datatype == 'amp' and eval:
        return transforms.Normalize(mean=VAL_AMP_MEAN, std=VAL_AMP_STD)(p.float()), \
            transforms.Normalize(mean=VAL_AMP_MEAN, std=VAL_AMP_STD)(gt.float())

    elif datatype == 'phase' and eval:
        return transforms.Normalize(mean=VAL_PHASE_MEAN, std=VAL_PHASE_STD)(p.float()), \
            transforms.Normalize(mean=VAL_PHASE_MEAN, std=VAL_PHASE_STD)(gt.float())

    elif datatype == 'amp' and not eval:
        return transforms.Normalize(mean=TRAIN_AMP_MEAN, std=TRAIN_AMP_STD)(p.float()), \
            transforms.Normalize(mean=TRAIN_AMP_MEAN, std=TRAIN_AMP_STD)(gt.float())

    elif datatype == 'phase' and not eval:
        return transforms.Normalize(mean=TRAIN_PHASE_MEAN, std=TRAIN_PHASE_STD)(p.float()), \
            transforms.Normalize(mean=TRAIN_PHASE_MEAN, std=TRAIN_PHASE_STD)(gt.float())
