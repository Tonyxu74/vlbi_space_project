import torch
import numpy as np
import matplotlib.pyplot as plt
from myargs import args
from utils.dataset import GenerateIterator, GenerateIterator_train, GenerateIterator_val
import segmentation_models_pytorch as smp
import tqdm
import time


def train():

    # define model and initialize weights if required
    def activation(x):
        x
    model_amp = eval('smp.'+args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model_amp.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model_phase = eval('smp.'+args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model_phase.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if args.continueTrain:
        pretrained_dict = torch.load('PRETRAINED AMP MODEL PATH HERE')['state_dict']
        model_dict = model_amp.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model_amp.load_state_dict(model_dict)

        pretrained_dict = torch.load('PRETRAINED PHASE MODEL PATH HERE')['state_dict']
        model_dict = model_phase.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model_phase.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(
        list(model_amp.parameters()) + list(model_phase.parameters()),
        lr=args.lr,
        weight_decay=args.weightDecay,
        betas=(args.beta1, args.beta2)
    )

    lossfn = torch.nn.MSELoss()

    iterator_train = GenerateIterator_train(args, './data/arrays/train', datatype='comb')
    iterator_val = GenerateIterator_val(args, './data/arrays/val', datatype='comb')

    if torch.cuda.is_available():
        model_amp = model_amp.cuda()
        model_phase = model_phase.cuda()
        lossfn = lossfn.cuda()

    start_epoch = 1

    for epoch in range(start_epoch, args.numEpochs):

        loss_sum, batch_num = 0, 0
        progress_bar = tqdm.tqdm(iterator_train, disable=False)
        start = time.time()

        if args.uvGenerate:
            num = args.numEpochs-epoch+1
            if num < 10:
                num = 10
            iterator_train.dataset.generate_uv(tele_num=num)

        for amp_images, phase_images, amp_gt, phase_gt in progress_bar:
            if torch.cuda.is_available():
                amp_images, amp_gt = amp_images.cuda(), amp_gt.cuda()
                phase_images, phase_gt = phase_images.cuda(), phase_gt.cuda()

            amp_prediction = model_amp(amp_images)
            phase_prediction = model_phase(phase_images)

            if not args.fftComb:
                loss_amp = lossfn(amp_prediction, amp_gt).mean()
                loss_phase = lossfn(phase_prediction, phase_gt).mean()

                loss = loss_amp + loss_phase

            else:
                real_prediction = torch.cos(phase_prediction) * amp_prediction
                im_prediction = torch.sin(phase_prediction) * amp_prediction

                real_gt = torch.cos(phase_gt) * amp_gt
                im_gt = torch.sin(phase_gt) * amp_gt

                # unnormalize PRED & GT???????
                complex_prediction = torch.cat((real_prediction.unsqueeze(4), im_prediction.unsqueeze(4)), dim=4)
                fft_prediction = torch.fft(complex_prediction, 3, normalized=False)

                complex_gt = torch.cat((real_gt.unsqueeze(4), im_gt.unsqueeze(4)), dim=4)
                fft_gt = torch.fft(complex_gt, 3, normalized=False)

                loss = lossfn(fft_prediction, fft_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f} '.format(loss_sum / (batch_num + 1e-6)))

        if epoch % 1 == 0:
            with torch.no_grad():
                model_amp.eval()
                model_phase.eval()

                # calculate accuracy on validation set
                if not args.fftComb:
                    amp_preds, amp_gts = [], []
                    phase_preds, phase_gts = [], []
                else:
                    fft_preds = [], fft_gts = []

                progress_bar = tqdm.tqdm(iterator_val)
                val_loss = 0
                for amp_images, phase_images, amp_gt, phase_gt in progress_bar:
                    if torch.cuda.is_available():
                        amp_images, amp_gt = amp_images.cuda(), amp_gt.cuda()
                        phase_images, phase_gt = phase_images.cuda(), phase_gt.cuda()

                    amp_prediction = model_amp(amp_images)
                    phase_prediction = model_phase(phase_images)

                    if not args.fftComb:
                        amp_preds.extend(amp_prediction.cpu().data.numpy())
                        amp_gts.extend(amp_gt.cpu().data.numpy())
                        phase_preds.extend(phase_prediction.cpu().data.numpy())
                        phase_gts.extend(phase_gt.cpu().data.numpy())

                    else:
                        real_prediction = torch.cos(phase_prediction) * amp_prediction
                        im_prediction = torch.sin(phase_prediction) * amp_prediction

                        real_gt = torch.cos(phase_gt) * amp_gt
                        im_gt = torch.sin(phase_gt) * amp_gt

                        # unnormalize PRED & GT???????
                        complex_prediction = torch.cat((real_prediction.unsqueeze(4), im_prediction.unsqueeze(4)), dim=4)
                        fft_prediction = torch.fft(complex_prediction, 3, normalized=False)

                        complex_gt = torch.cat((real_gt.unsqueeze(4), im_gt.unsqueeze(4)), dim=4)
                        fft_gt = torch.fft(complex_gt, 3, normalized=False)

                        fft_preds.extend(fft_prediction.cpu().data.numpy())
                        fft_gts.extend(fft_gt.cpu().data.numpy())
                if not args.fftComb:
                    amp_preds = np.asarray(amp_preds)
                    amp_gts = np.asarray(amp_gts)
                    phase_preds = np.asarray(phase_preds)
                    phase_gts = np.asarray(phase_gts)

                    val_f1_score_amp = (np.sum(np.abs(amp_preds.flatten() - amp_gts.flatten())) / len(amp_gts.flatten()))
                    val_f1_score_phase = (np.sum(np.abs(phase_preds.flatten() - phase_gts.flatten())) / len(phase_gts.flatten()))

                    print(
                        '|| Ep {} || Secs {:.1f} || Loss {:.1f} || Val f1 score amp {:.3f} || Val f1 score phase {:.3f} Val Loss {:.3f} ||\n'.format(
                            epoch,
                            time.time() - start,
                            loss_sum / batch_num,
                            val_f1_score_amp,
                            val_f1_score_phase,
                            val_loss,
                        ))

                else:
                    fft_preds = np.asarray(fft_preds)
                    fft_gts = np.asarray(fft_gts)

                    val_f1_score_fft = (np.sum(np.abs(fft_preds.flatten() - fft_gts.flatten())) / len(fft_gts.flatten()))

                    print(
                        '|| Ep {} || Secs {:.1f} || Loss {:.1f} || Val f1 score {:.3f} || Val Loss {:.3f} ||\n'.format(
                            epoch,
                            time.time() - start,
                            loss_sum / batch_num,
                            val_f1_score_fft,
                            val_loss,
                        ))

        if epoch % 5 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model_amp.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/models/amp/comb_model_{}_{}.pt'.format(args.modelName, epoch))

            state = {
                'epoch': epoch,
                'state_dict': model_phase.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/models/phase/comb_model_{}_{}.pt'.format(args.modelName, epoch))


if __name__ == '__main__':
    train()
