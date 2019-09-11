import numpy as np
import torch
from PIL import Image
from utils.dataset import GenerateIterator
from myargs import args
import segmentation_models_pytorch as smp

datatype = 'phase'


def visualize_val(epoch):
    iterator_val = GenerateIterator(args, '../data/arrays/val', eval=True, datatype=datatype)

    # model definition
    def activation(x):
        x
    model = eval('smp.'+args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model = model.cuda()

    # load weights
    pretrained_dict = torch.load('../data/models/{}/model_Unet_{}.pt'.format(datatype, epoch))['state_dict']
    model_dict = model.state_dict()

    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    with torch.no_grad():
        model.eval()

        image_num = 0

        for images, gt in iterator_val:
            if torch.cuda.is_available():
                images = images.cuda()
            prediction = model(images)

            for pred_visibility, gt_visibility in zip(prediction, gt):
                pred_visibility = pred_visibility.cpu().data.numpy()
                pred_brightness = np.fft.ifft2(pred_visibility)
                pred_brightness = np.fft.ifftshift(pred_brightness)
                pred_brightness = np.abs(pred_brightness)

                gt_brightness = np.fft.ifft2(gt_visibility.cpu().data.numpy())
                #gt_brightness = np.fft.ifftshift(gt_brightness)
                gt_brightness = np.abs(gt_brightness)

                max_brightness = np.max(pred_brightness)
                brightness_image = Image.fromarray((255 / max_brightness * pred_brightness).astype(np.uint8)[0])
                brightness_image.save('../data/out/{}_prediction.png'.format(image_num))

                max_brightness_gt = np.max(gt_brightness)
                brightness_gt = Image.fromarray((255 / max_brightness_gt * gt_brightness).astype(np.uint8)[0])
                brightness_gt.save('../data/out/{}_gt.png'.format(image_num))

                image_num += 1


def visualize_comb(epoch_amp, epoch_phase):
    print('hi')


if __name__ == "__main__":
    visualize_val(4)
