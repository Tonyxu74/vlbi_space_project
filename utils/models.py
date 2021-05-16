import numpy as np
import torch
from torch import nn
from myargs import args
from torchvision.models import resnet18


'''
Implementing https://arxiv.org/ftp/arxiv/papers/1704/1704.08841.pdf
AUTOMAP for k-space reconstructions
let's give it a try!

model intakes the flattened complex uv plane data, so that it's (Images x 2*64^2) dimension vector
the output is the reconstruction AFTER the fourier transform, meaning it is the reconstructed magnitude image, and can
simply be MSE error'd with the iFFT of the frequency GT
'''

class AUTOMAP_Model(nn.Module):
    def __init__(self):
        super(AUTOMAP_Model, self).__init__()

        '''
        -> Fully connected (FC1) -> tanh activation: input/output size (n_H0 * n_W0 * 2, n_H0 * n_W0) 
        -> Fully connected (FC2) -> tanh activation: input/output size (n_H0 * n_W0, n_H0 * n_W0) 
        -> Convolutional -> ReLU activation: size (n_im, n_H0, n_W0, 64)
        -> Convolutional -> ReLU activation: size (n_im, n_H0, n_W0, 64)
        -> De-convolutional: size (n_im, n_H0, n_W0)
        '''
        self.input_size = args.imageDims[0] * args.imageDims[1] * 2  # complex image of 64x64 with Re and Im separate
        self.FC1 = nn.Linear(self.input_size, self.input_size // 2)
        self.FC2 = nn.Linear(self.input_size // 2, self.input_size // 2)
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.Conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(7, 7), stride=1, padding=3)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.FC1(x)
        x = self.tanh(x)

        x = self.FC2(x)
        x = self.tanh(x)

        x = torch.reshape(x, (-1, 1, args.imageDims[0], args.imageDims[1]))  # (Batch, 1, H, W)

        x = self.Conv1(x)
        x = self.relu(x)

        x = self.Conv2(x)
        x = self.relu(x)

        x = self.Conv3(x)

        return x


class Discriminator(nn.Module):
    """
    Simple discriminator to determine if image is generated or original image
    """
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        def activation(x):
            x

        model = resnet18(pretrained=True)

        self.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x
