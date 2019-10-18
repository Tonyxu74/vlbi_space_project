import torch
import numpy as np
import matplotlib.pyplot as plt
from myargs import args
from utils.dataset import GenerateIterator, GenerateIterator_train, GenerateIterator_val
import segmentation_models_pytorch as smp
import tqdm
import time


def train(datatype):

    # define model and initialize weights if required
    def activation(x):
        x
    model = eval('smp.'+args.modelName)(
        args.encoderName,
        encoder_weights='imagenet',
        classes=1,
        activation=activation,
    )

    model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if args.continueTrain:
        pretrained_dict = torch.load('PRETRAINED MODEL PATH HERE')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weightDecay,
        betas=(args.beta1, args.beta2)
    )

    lossfn = torch.nn.MSELoss()

    iterator_train = GenerateIterator_train(args, './data/arrays/train', datatype=datatype)
    iterator_val = GenerateIterator_val(args, './data/arrays/val', datatype=datatype)

    if torch.cuda.is_available():
        model = model.cuda()
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

        for images, gt in progress_bar:
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            prediction = model(images)

            loss = lossfn(prediction, gt).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batch_num += 1

            progress_bar.set_description('Loss: {:.5f} '.format(loss_sum / (batch_num + 1e-6)))

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                # calculate accuracy on validation set
                preds, gts = [], []
                progress_bar = tqdm.tqdm(iterator_val)
                val_loss = 0
                for images, gt in progress_bar:
                    if torch.cuda.is_available():
                        images, gt = images.cuda(), gt.cuda()

                    prediction = model(images)

                    val_loss += lossfn(prediction, gt).mean()

                    preds.extend(prediction.cpu().data.numpy())
                    gts.extend(gt.cpu().data.numpy())

                preds = np.asarray(preds)
                gts = np.asarray(gts)

                val_f1_score = (np.sum(np.abs(preds.flatten() - gts.flatten())) / len(gts.flatten()))

            print('|| Ep {} || Secs {:.1f} || Loss {:.1f} || Val f1 score {:.3f} || Val Loss {:.3f} ||\n'.format(
                    epoch,
                    time.time() - start,
                    loss_sum,
                    val_f1_score,
                    val_loss,
                )
            )
        if epoch % 5 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './data/models/{}/model_{}_{}.pt'.format(datatype, args.modelName, epoch))


if __name__ == '__main__':
    train('amp')
    train('phase')
