import numpy as np
import torch
from PIL import Image
from utils.dataset import GenerateIterator
from myargs import args
import segmentation_models_pytorch as smp

VAL_AMP_MEAN = 0.00221
VAL_AMP_STD = 0.03018

VAL_PHASE_MEAN = 0.
VAL_PHASE_STD = 1.8104


def visualize_val(epoch):
    datatype = 'phase'
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
                pred_visibility = pred_visibility * VAL_PHASE_STD + VAL_PHASE_MEAN
                pred_brightness = np.fft.ifft2(pred_visibility)
                pred_brightness = np.fft.ifftshift(pred_brightness)
                pred_brightness = np.abs(pred_brightness)
                pred_brightness[:, 120:136, 120:136] = 0

                gt_brightness = np.fft.ifft2(gt_visibility.cpu().data.numpy() * VAL_PHASE_STD + VAL_PHASE_MEAN)
                gt_brightness = np.fft.ifftshift(gt_brightness)
                gt_brightness = np.abs(gt_brightness)
                gt_brightness[:, 120:136, 120:136] = 0

                max_brightness = np.max(pred_brightness)
                brightness_image = Image.fromarray((255 / max_brightness * pred_brightness).astype(np.uint8)[0])
                brightness_image.save('../data/out/{}_prediction.png'.format(image_num))

                max_brightness_gt = np.max(gt_brightness)
                brightness_gt = Image.fromarray((255 / max_brightness_gt * gt_brightness).astype(np.uint8)[0])
                brightness_gt.save('../data/out/{}_gt.png'.format(image_num))

                image_num += 1


def visualize_comb(epoch_amp, epoch_phase):
    iterator_val = GenerateIterator(args, '../data/arrays/val', eval=True, datatype='comb')

    # model definition
    def activation(x):
        x

    # amplitude model ==============================
    model_amp = eval('smp.' + args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model_amp.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model_amp = model_amp.cuda()

    # phase model ==============================
    model_phase = eval('smp.' + args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model_phase.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model_phase = model_phase.cuda()

    # amp model ==============================
    # load weights
    pretrained_dict = torch.load('../data/models/{}/model_Unet_{}.pt'.format('amp', epoch_amp))['state_dict']
    model_dict = model_amp.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model_amp.load_state_dict(model_dict)

    # phase model ==============================
    # load weights
    pretrained_dict = torch.load('../data/models/{}/model_Unet_{}.pt'.format('phase', epoch_phase))['state_dict']
    model_dict = model_phase.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model_phase.load_state_dict(model_dict)

    with torch.no_grad():
        model_amp.eval()
        model_phase.eval()

        image_num = 0

        for amp_images, phase_images, amp_gt, phase_gt in iterator_val:
            if torch.cuda.is_available():
                amp_images = amp_images.cuda()
                phase_images = phase_images.cuda()

            pred_amp = model_amp(amp_images)
            pred_phase = model_phase(phase_images)

            for pred_vis_amp, pred_vis_phase, gt_vis_amp, gt_vis_phase in zip(pred_amp, pred_phase, amp_gt, phase_gt):

                pred_vis_amp = pred_vis_amp.cpu().data.numpy() * VAL_AMP_STD + VAL_AMP_MEAN
                pred_vis_phase = pred_vis_phase.cpu().data.numpy() * VAL_PHASE_STD + VAL_PHASE_MEAN
                comb_gt = np.multiply(pred_vis_amp, np.exp(1j * pred_vis_phase))
                pred_brightness = np.fft.ifft2(comb_gt)
                pred_brightness = np.fft.ifftshift(pred_brightness)
                pred_brightness = np.abs(pred_brightness)
                pred_brightness[:, 115:141, 115:141] = 0

                gt_vis_amp = gt_vis_amp.cpu().data.numpy() * VAL_AMP_STD + VAL_AMP_MEAN
                gt_vis_phase = gt_vis_phase.cpu().data.numpy() * VAL_PHASE_STD + VAL_PHASE_MEAN
                comb_gt = np.multiply(gt_vis_amp, np.exp(1j * gt_vis_phase))
                gt_brightness = np.fft.ifft2(comb_gt)
                # gt_brightness = np.fft.ifftshift(gt_brightness)
                gt_brightness = np.abs(gt_brightness)

                max_brightness = np.max(pred_brightness)
                brightness_image = Image.fromarray((255 / max_brightness * pred_brightness).astype(np.uint8)[0])
                brightness_image.save('../data/out/{}_prediction.png'.format(image_num))

                max_brightness_gt = np.max(gt_brightness)
                brightness_gt = Image.fromarray((255 / max_brightness_gt * gt_brightness).astype(np.uint8)[0])
                brightness_gt.save('../data/out/{}_gt.png'.format(image_num))

                image_num += 1


if __name__ == "__main__":
    visualize_comb(45, 45)
