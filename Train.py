import sys

import torch.nn as nn
import torch
import tqdm

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)


def dice_loss(predict, target):
    smooth = 1.
    loss = 0.
    for c in range(predict.shape[1]):
        iflat = predict[:, c, ...].contiguous().view(-1)
        tflat = target[:, c, ...].contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        loss +=  (1 - ((2. * intersection + smooth) /
                          (iflat.sum() + tflat.sum() + smooth)))
    return loss

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        loss_func = nn.BCELoss()
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        loss = 0
        for b_idx in range(predict.shape[0]):
            for row in range(predict.shape[2]):
                for col in range(predict.shape[3]):
                    predict_i = predict[b_idx,:,row,col].contiguous().view(-1)
                    target_i = target[b_idx,:,row,col].contiguous().view(-1)
                    loss += loss_func(predict_i, target_i)
                    # num = torch.sum(torch.mul(predict_i, target_i), dim=0) + self.smooth
                    # den = torch.sum(predict_i.pow(self.p) + target_i.pow(self.p), dim=0) + self.smooth
                    #
                    # loss = loss +  1 - num / den

        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)
        #
        # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        #
        # loss = 1 - num / den
        return loss
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'none':
        #     return loss
        # else:
        #     raise Exception('Unexpected reduction {}'.format(self.reduction))

def eval_loss_epoch(training_loader, model, loss_function):

    losses = []
    model.eval()
    with tqdm.tqdm(total=len(training_loader), file=sys.stdout) as pbar:
        for idx_batch, (images, masks) in enumerate(training_loader, start=1):

            images = images.to(device)
            masks = masks.to(device)

            # calculate output
            y_hat = model(images)

            # calculate loss now:

            loss = loss_function(y_hat, masks)

            # optimizing weights

            # update loss bar
            losses.append(loss.detach())
            pbar.update();
            pbar.set_description(f'val loss={losses[-1]:.3f}')
        mean_loss = torch.mean(torch.FloatTensor(losses))
        pbar.set_description(f'val loss={mean_loss:.3f}')

    return [mean_loss]

def train_epoch(training_loader, model, optimizer, loss_function):

    losses = []
    model.train()
    with tqdm.tqdm(total=len(training_loader), file=sys.stdout) as pbar:
        for idx_batch, (images, masks) in enumerate(training_loader, start=1):

            images = images.to(device)
            masks = masks.to(device)

            # calculate output
            y_hat = model(images)

            # calculate loss now:
            optimizer.zero_grad()
            loss = loss_function(y_hat, masks)
            loss.backward()

            # optimizing weights
            optimizer.step()

            # update loss bar
            losses.append(loss.detach())
            pbar.update();
            pbar.set_description(f'train loss={losses[-1]:.3f}')
        mean_loss = torch.mean(torch.FloatTensor(losses))
        pbar.set_description(f'train loss={mean_loss:.3f}')

    return [mean_loss]
