import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--modelName', type=str, default='Unet',
                    help='reconstruction model name')
parser.add_argument('--encoderName', type=str, default='resnet50',
                    help='encoder model name')
parser.add_argument('--batchSize', type=int,
                    default=128, help='input batch size')
parser.add_argument('--trainNum', type=int,
                    default=10000, help='input batch size')
parser.add_argument('--numEpochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--warmupEpochs', type=int, default=10,
                    help='number of warmup epochs for scheduler')
parser.add_argument('--continueEpoch', type=int, default=51,
                    help='epoch to continue training from')
parser.add_argument('--continueTrain', type=bool, default=False,
                    help='continue to train from saved dict')
parser.add_argument('--imageDims', type=tuple,
                    default=(96, 96), help='dimensions of image')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers to use from cpu')
parser.add_argument('--uvGenerate', type=bool, default=True,
                    help='Use the UV generator train?')
parser.add_argument('--fftComb', type=bool, default=True,
                    help='train by combining with fft?')
parser.add_argument('--unnormalize', type=bool, default=True,
                    help='Unnormalize weights before FFT?')

parser.add_argument('--maxradWarmup', type=tuple,
                    default=(0.1, 0.3), help='maxrad warmup range')
parser.add_argument('--maxradWarmupEpochs', type=tuple,
                    default=20, help='maxrad warmup epoch num')

parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--weightDecay', type=float, default=0.05,
                    help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta 1 value for optim')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta 2 value for optim')

args = parser.parse_args()
