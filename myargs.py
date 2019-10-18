import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--modelName', type=str, default='Unet',
                    help='input batch size')
parser.add_argument('--encoderName', type=str, default='resnet18',
                    help='input batch size')
parser.add_argument('--batchSize', type=int,
                    default=8, help='input batch size')
parser.add_argument('--trainNum', type=int,
                    default=10000, help='input batch size')
parser.add_argument('--numEpochs', type=int, default=50,
                    help='number of epochs to train for')
parser.add_argument('--continueTrain', type=bool, default=False,
                    help='continue to train from saved dict')
parser.add_argument('--imageDims', type=tuple,
                    default=(64, 64), help='dimensions of image')
parser.add_argument('--workers', type=int, default=0,
                    help='number of workers to use from cpu')
parser.add_argument('--uvGenerate', type=bool, default=True,
                    help='Use the UV generator train?')
parser.add_argument('--fftComb', type=bool, default=True,
                    help='train by combining with fft?')
parser.add_argument('--unnormalize', type=bool, default=True,
                    help='Unnormalize weights before FFT?')

parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weightDecay', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta 1 value for optim')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta 2 value for optim')

args = parser.parse_args()
