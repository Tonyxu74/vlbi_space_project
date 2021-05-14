import torch
import numpy as np
import matplotlib.pyplot as plt
from myargs import args
from utils.dataset import GenerateIterator, GenerateIterator_train, GenerateIterator_val
import segmentation_models_pytorch as smp
from utils.models import Discriminator
import tqdm
import time

TRAIN_AMP_MEAN = (16.9907,)
TRAIN_AMP_STD = (157.3094,)

TRAIN_PHASE_MEAN = (0.,)
TRAIN_PHASE_STD = (1.8141,)

VAL_AMP_MEAN = (0.00221,)
VAL_AMP_STD = (0.03018,)

VAL_PHASE_MEAN = (0.,)
VAL_PHASE_STD = (1.8104,)

'''CONSIDER CHANGING TORCH FFT TO TORCH IFFT CUZ TECHNICALLY WE ARE IN THE UV PLANE WE IFFT TO GET TO NORMAL PLANE'''
'''tried ifft, didn't work too well as weights went down too much? try training longer (still didn't work oops)'''


def train():

    # define model and initialize weights for amplitude
    def activation(x):
        x
    model_amp = eval('smp.'+args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model_amp.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # define model and initialize weights for phase
    model_phase = eval('smp.'+args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )
    model_phase.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model_disc = Discriminator()

    # check if continue training from previous epochs
    if args.continueTrain:
        pretrained_dict = torch.load('./data/models/amp/comb_model_Unet_{}.pt'.format(args.continueEpoch))['state_dict']
        model_dict = model_amp.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model_amp.load_state_dict(model_dict)

        pretrained_dict = torch.load('./data/models/phase/comb_model_Unet_{}.pt'.format(args.continueEpoch))['state_dict']
        model_dict = model_phase.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model_phase.load_state_dict(model_dict)

    # define reconstruction + discriminator optimizer, loss function, and iterators
    optimizer_rec = torch.optim.Adam(
        list(model_amp.parameters()) + list(model_phase.parameters()),
        lr=args.lr,
        weight_decay=args.weightDecay,
        betas=(args.beta1, args.beta2)
    )
    optimizer_disc = torch.optim.Adam(
        list(model_disc.parameters()),
        lr=args.lr,
        weight_decay=args.weightDecay,
        betas=(args.beta1, args.beta2)
    )

    # pixel-by-pixel and adversarial losses
    lossfn = torch.nn.MSELoss()
    lossfn_adv = torch.nn.BCELoss()

    iterator_train = GenerateIterator_train(args, './data/arrays/train', datatype='comb')
    iterator_val = GenerateIterator_val(args, './data/arrays/val', datatype='comb')

    # cuda?
    if torch.cuda.is_available():
        model_amp = model_amp.cuda()
        model_phase = model_phase.cuda()
        model_disc = model_disc.cuda()
        lossfn = lossfn.cuda()
        lossfn_adv = lossfn_adv.cuda()

    start_epoch = 1

    for epoch in range(start_epoch, args.numEpochs):

        # values to look at average loss per batch over epoch
        rec_loss_sum, disc_loss_sum, batch_num = 0, 0, 0
        progress_bar = tqdm.tqdm(iterator_train, disable=False)
        start = time.time()

        # generates a UV plane with "num" number of telescopes
        # (the more there are, the easier the problem is for the model)
        # trying with fixed 10, but maybe try annealing
        if args.uvGenerate:
            num = args.numEpochs-epoch+1
            if num < 10:
                num = 10

            iterator_train.dataset.generate_uv(tele_num=num)

        '''======== TRAIN ========'''
        for amp_images, phase_images, amp_gt, phase_gt in progress_bar:
            rec_image_labels = torch.ones(amp_images.shape[0],)
            real_image_labels = torch.zeros(amp_images.shape[0],)

            if torch.cuda.is_available():
                amp_images, amp_gt = amp_images.cuda(), amp_gt.cuda()
                phase_images, phase_gt = phase_images.cuda(), phase_gt.cuda()
                rec_image_labels = rec_image_labels.cuda()
                real_image_labels = real_image_labels.cuda()

            # ----- get amp/phase reconstructions -----
            amp_prediction = model_amp(amp_images)
            phase_prediction = model_phase(phase_images)

            loss_amp = lossfn(amp_prediction, amp_gt).mean()
            loss_phase = lossfn(phase_prediction, phase_gt).mean()

            # ----- get brightness images ------
            # discriminator acts on brightness image combining phase and amp models with a complex fft
            # Re(Z) = |Z| * cos(phi)
            real_prediction = torch.cos(phase_prediction * VAL_PHASE_STD[0]) * amp_prediction
            # Im(Z) = |Z| * sin(phi)
            im_prediction = torch.sin(phase_prediction * VAL_PHASE_STD[0]) * amp_prediction

            real_gt = torch.cos(phase_gt * VAL_PHASE_STD[0]) * amp_gt
            im_gt = torch.sin(phase_gt * VAL_PHASE_STD[0]) * amp_gt

            # this section concatenates the real and imaginary predictions properly for torch.fft
            # perhaps change 3 dims to 2 dims
            complex_prediction = torch.cat((real_prediction.unsqueeze(4), im_prediction.unsqueeze(4)), dim=4)
            fft_prediction = torch.ifft(complex_prediction, signal_ndim=3, normalized=True)

            complex_gt = torch.cat((real_gt.unsqueeze(4), im_gt.unsqueeze(4)), dim=4)
            fft_gt = torch.ifft(complex_gt, signal_ndim=3, normalized=True)

            # ----- train reconstruction model -----
            # add reconstruction loss and mse losses
            rec_loss = lossfn_adv(model_disc(fft_prediction), real_image_labels) + loss_amp + loss_phase

            optimizer_rec.zero_grad()
            rec_loss.backward()
            optimizer_rec.step()

            rec_loss_sum += rec_loss.item()

            # ----- train discriminator model -----
            # divide discriminator loss by 2 so that loss weights are the same
            disc_loss = (lossfn_adv(model_disc(fft_prediction.detach()), rec_image_labels) +
                         lossfn_adv(model_disc(fft_gt), real_image_labels)) / 2

            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()

            disc_loss_sum += disc_loss.item()
            batch_num += 1

            progress_bar.set_description('Rec Loss: {:.5f} || Disc Loss {:.5f}'.format(
                rec_loss_sum / (batch_num + 1e-6), disc_loss_sum / (batch_num + 1e-6)
            ))

        '''======== VALIDATION ========'''
        if epoch % 1 == 0:
            with torch.no_grad():
                model_amp.eval()
                model_phase.eval()

                amp_preds, amp_gts = [], []
                phase_preds, phase_gts = [], []

                progress_bar = tqdm.tqdm(iterator_val)
                val_loss = 0

                for amp_images, phase_images, amp_gt, phase_gt in progress_bar:
                    if torch.cuda.is_available():
                        amp_images, amp_gt = amp_images.cuda(), amp_gt.cuda()
                        phase_images, phase_gt = phase_images.cuda(), phase_gt.cuda()

                    # just verify MSE error for validation
                    amp_prediction = model_amp(amp_images)
                    phase_prediction = model_phase(phase_images)

                    amp_preds.extend(amp_prediction.cpu().data.numpy())
                    amp_gts.extend(amp_gt.cpu().data.numpy())
                    phase_preds.extend(phase_prediction.cpu().data.numpy())
                    phase_gts.extend(phase_gt.cpu().data.numpy())

                    # MSE loss from difference in freq domain
                    loss = lossfn(amp_prediction, amp_gt).mean() + lossfn(phase_prediction, phase_gt).mean()

                    val_loss += loss.item()

                amp_preds = np.asarray(amp_preds)
                amp_gts = np.asarray(amp_gts)
                phase_preds = np.asarray(phase_preds)
                phase_gts = np.asarray(phase_gts)

                val_f1_score_amp = (np.sum(np.abs(amp_preds.flatten() - amp_gts.flatten())) / len(amp_gts.flatten()))
                val_f1_score_phase = (np.sum(np.abs(phase_preds.flatten() - phase_gts.flatten())) / len(phase_gts.flatten()))

                print(
                    '|| Ep {} || Secs {:.1f} || Rec Loss {:.3f} || Disc Loss {:.3f} || Val f1 score amp {:.3f} || Val f1 score phase {:.3f} Val Loss {:.3f} ||\n'.format(
                        epoch,
                        time.time() - start,
                        rec_loss_sum,
                        disc_loss_sum,
                        val_f1_score_amp,
                        val_f1_score_phase,
                        val_loss,
                    ))

            model_amp.train()
            model_phase.train()

        # save models every epoch
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model_amp.state_dict(),
            }
            torch.save(state, './data/models/amp/comb_model_{}_{}.pt'.format(args.modelName, epoch))

            state = {
                'epoch': epoch,
                'state_dict': model_phase.state_dict(),
            }
            torch.save(state, './data/models/phase/comb_model_{}_{}.pt'.format(args.modelName, epoch))

            state = {
                'epoch': epoch,
                'state_dict': model_disc.state_dict(),
            }
            torch.save(state, './data/models/disc/comb_model_{}_{}.pt'.format(args.modelName, epoch))


if __name__ == '__main__':
    train()
