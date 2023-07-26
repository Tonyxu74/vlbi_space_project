import torch
from myargs import args
from utils.dataset_img import GenerateTrainImageIterator, GenerateValidImageIterator
from utils.lrscheduler import WarmupCosineScheduler
import segmentation_models_pytorch as smp
import tqdm
import time
from utils.pytorch_ssim import SSIM


def train():

    # define model and initialize weights
    model = eval('smp.' + args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        in_channels=2,
        activation=None
    )

    # check if continue training from previous epochs
    if args.continueTrain:
        pretrained_dict = torch.load('./data/models/img_domain/img_model_Unet_{}.pt'.format(args.continueEpoch))['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f'Successfully loaded weights from epoch {args.continueEpoch}.')

    # define optimizer, loss function, and iterators
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weightDecay,
        betas=(args.beta1, args.beta2)
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmupEpochs,
        max_lr=args.lr,
        max_epochs=args.numEpochs,
        min_lr=args.lr / args.warmupEpochs
    )

    lossfn = torch.nn.L1Loss()
    ssim_fn = SSIM()

    iterator_train = GenerateTrainImageIterator(args, './raw_data/256_ObjectCategories')
    iterator_val = GenerateValidImageIterator(args, './data/img_arrays')

    # cuda?
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()
        ssim_fn = ssim_fn.cuda()

    start_epoch = 1

    for epoch in range(start_epoch, args.numEpochs):

        # values to look at average loss per batch over epoch
        loss_sum, batch_num, ssim_sum = 0, 0, 0
        progress_bar = tqdm.tqdm(iterator_train, disable=False)
        start = time.time()

        # update upper maxrad between 0.1 and 0.3 from epoch 1 to 21
        if (epoch - 1) < args.maxradWarmupEpochs:
            upper_maxrad = (epoch - 1) / args.maxradWarmupEpochs * \
                           (args.maxradWarmup[1] - args.maxradWarmup[0]) + args.maxradWarmup[0]
        else:
            upper_maxrad = args.maxradWarmup[1]
        iterator_train.dataset.maxrad_range[1] = upper_maxrad
        print('Learning rate: {:.6f} || Maxrad range: {} ||'.format(
            optimizer.param_groups[0]['lr'], iterator_train.dataset.maxrad_range))

        '''======== TRAIN ========'''
        for dirty_input, target_img in progress_bar:
            if torch.cuda.is_available():
                dirty_input, target_img = dirty_input.cuda(), target_img.cuda()

            pred_recon = model(dirty_input)
            l1loss = lossfn(pred_recon, target_img).mean()
            ssim = ssim_fn(pred_recon, target_img)
            loss = l1loss + (1 - ssim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            ssim_sum += ssim.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f}, SSIM: {:.5f}'.format(loss.item(), ssim.item()))

        '''======== VALIDATION ========'''
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                progress_bar = tqdm.tqdm(iterator_val)
                val_loss, val_batch, val_ssim = 0, 0, 0

                for dirty_input, target_img in progress_bar:
                    if torch.cuda.is_available():
                        dirty_input, target_img = dirty_input.cuda(), target_img.cuda()

                    pred_recon = model(dirty_input)
                    l1loss = lossfn(pred_recon, target_img).mean()
                    ssim = ssim_fn(pred_recon, target_img)
                    loss = l1loss + (1 - ssim)

                    val_loss += loss.item()
                    val_ssim += ssim.item()

                    val_batch += 1

                print(
                    '|| Ep {} || Secs {:.1f} || Loss {:.3f} || Val Loss {:.3f} || SSIM {:.3f} || Val SSIM {:.3f}'.format(
                        epoch,
                        time.time() - start,
                        loss_sum / batch_num,
                        val_loss / val_batch,
                        ssim_sum / batch_num,
                        val_ssim / val_batch
                    ))

            model.train()

        # update learning rate
        scheduler.step()

        # save models every 1 epoch
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict()
            }
            torch.save(state, './data/models/img_domain/img_model_{}_{}.pt'.format(args.modelName, epoch))


if __name__ == '__main__':
    train()
