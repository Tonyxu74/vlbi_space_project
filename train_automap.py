import torch
import numpy as np
from myargs import args
from utils.dataset import GenerateIterator_train, GenerateIterator_val
import tqdm
import time
from utils.automap_model import AUTOMAP_Model

TRAIN_AMP_MEAN = (16.9907,)
TRAIN_AMP_STD = (157.3094,)

TRAIN_PHASE_MEAN = (0.,)
TRAIN_PHASE_STD = (1.8141,)

VAL_AMP_MEAN = (0.00221,)
VAL_AMP_STD = (0.03018,)

VAL_PHASE_MEAN = (0.,)
VAL_PHASE_STD = (1.8104,)


def train():

    model = AUTOMAP_Model()

    # check if continue training from previous epochs
    if args.continueTrain:
        pretrained_dict = torch.load('./data/models/amp/comb_model_Unet_{}.pt'.format(args.continueEpoch))['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # define optimizer, loss function, and iterators
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weightDecay,
        betas=(args.beta1, args.beta2)
    )

    lossfn = torch.nn.MSELoss()

    iterator_train = GenerateIterator_train(args, './data/arrays/train', datatype='comb')
    iterator_val = GenerateIterator_val(args, './data/arrays/val', datatype='comb')

    # cuda?
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    start_epoch = 1

    for epoch in range(start_epoch, args.numEpochs):

        # values to look at average loss per batch over epoch
        loss_sum, batch_num = 0, 0
        progress_bar = tqdm.tqdm(iterator_train, disable=False)
        start = time.time()

        # generates a UV plane with "num" number of telescopes
        # (the more there are, the easier the problem is for the model)
        # trying with fixed 10, but maybe try annealing
        if args.uvGenerate:
            iterator_train.dataset.generate_uv(tele_num=10)

        '''======== TRAIN ========'''
        for amp_images, phase_images, amp_gt, phase_gt in progress_bar:
            if torch.cuda.is_available():
                amp_images, amp_gt = amp_images.cuda(), amp_gt.cuda()
                phase_images, phase_gt = phase_images.cuda(), phase_gt.cuda()

            # Re(Z) = |Z| * cos(phi)
            real_img = torch.cos(phase_images * VAL_PHASE_STD[0]) * amp_images
            # Im(Z) = |Z| * sin(phi)
            im_img = torch.sin(phase_images * VAL_PHASE_STD[0]) * amp_images

            real_gt = torch.cos(phase_gt * VAL_PHASE_STD[0]) * amp_gt
            im_gt = torch.sin(phase_gt * VAL_PHASE_STD[0]) * amp_gt

            # this section concatenates the real and imaginary predictions and flattens it
            complex_img = torch.cat((real_img.unsqueeze(4), im_img.unsqueeze(4)), dim=4)
            flat_img = complex_img.reshape(shape=(-1, args.imageDims[0] * args.imageDims[1] * 2))

            # use fft to get a complex label
            complex_gt = torch.cat((real_gt.unsqueeze(4), im_gt.unsqueeze(4)), dim=4)
            fft_gt = torch.ifft(complex_gt, signal_ndim=3, normalized=True)
            # now make it only real by taking the Re of it (Im amplitude should be small but maybe consider actually
            # finding magnitude with sqrt(a^2 + b^2) for z = a + bi
            fft_gt = torch.sqrt(fft_gt[..., 0] * fft_gt[..., 0] + fft_gt[..., 1] * fft_gt[..., 1])

            prediction = model(flat_img)

            loss = lossfn(prediction, fft_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f} '.format(loss_sum / (batch_num + 1e-6)))

        '''======== VALIDATION ========'''
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                preds, fft_gts = [], []

                progress_bar = tqdm.tqdm(iterator_val)
                val_loss = 0

                for amp_images, phase_images, amp_gt, phase_gt in progress_bar:
                    if torch.cuda.is_available():
                        amp_images, amp_gt = amp_images.cuda(), amp_gt.cuda()
                        phase_images, phase_gt = phase_images.cuda(), phase_gt.cuda()

                    # we unnormalize here but not in train because the nature of these images are different, one uses
                    # the actual vis data, the other uses the UV generated stuff
                    # Re(Z) = |Z| * cos(phi)
                    real_img = torch.cos(phase_images * VAL_PHASE_STD) * (amp_images * VAL_AMP_STD + VAL_AMP_MEAN)
                    # Im(Z) = |Z| * sin(phi)
                    im_img = torch.sin(phase_images * VAL_PHASE_STD) * (amp_images * VAL_AMP_STD + VAL_AMP_MEAN)

                    real_gt = torch.cos(phase_gt * VAL_PHASE_STD) * (amp_gt * VAL_AMP_STD + VAL_AMP_MEAN)
                    im_gt = torch.sin(phase_gt * VAL_PHASE_STD) * (amp_gt * VAL_AMP_STD + VAL_AMP_MEAN)

                    # this section concatenates the real and imaginary predictions and flattens it
                    complex_img = torch.cat((real_img.unsqueeze(4), im_img.unsqueeze(4)), dim=4)
                    flat_img = complex_img.reshape(shape=(-1, args.imageDims[0] * args.imageDims[1] * 2))

                    # use irfft to get a real-valued magnitude label
                    complex_gt = torch.cat((real_gt.unsqueeze(4), im_gt.unsqueeze(4)), dim=4)
                    fft_gt = torch.ifft(complex_gt, signal_ndim=3, normalized=True)
                    # now make it only real by taking the Re of it (Im amplitude should be small but maybe consider
                    # actually finding magnitude with sqrt(a^2 + b^2) for z = a + bi
                    fft_gt = torch.sqrt(fft_gt[..., 0] * fft_gt[..., 0] + fft_gt[..., 1] * fft_gt[..., 1])

                    prediction = model(flat_img)

                    preds.extend(prediction.cpu().data.numpy())
                    fft_gts.extend(fft_gt.cpu().data.numpy())

                    loss = lossfn(prediction, fft_gt).mean()

                    val_loss += loss.item()

                preds = np.asarray(preds)
                fft_gts = np.asarray(fft_gts)

                val_f1_score_fft = (np.sum(np.abs(preds.flatten() - fft_gts.flatten())) / len(fft_gts.flatten()))

                print(
                    '|| Ep {} || Secs {:.1f} || Loss {:.1f} || Val f1 score fft {:.3f} || Val Loss {:.3f} ||\n'.format(
                        epoch,
                        time.time() - start,
                        loss_sum,
                        val_f1_score_fft,
                        val_loss,
                    ))

            model.train()

        # save models every 1 epoch
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/models/automap_model/automap_model_{}.pt'.format(epoch))


if __name__ == '__main__':
    train()
