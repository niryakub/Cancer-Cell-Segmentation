import sys

import torch.nn as nn
import torch
import tqdm

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

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
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

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
#
# def train_model(model, training_loader, optimizer, loss_function, max_batches=None):
#     losses = []
#     with tqdm.tqdm(total=(max_batches if max_batches else len(training_loader)), file=sys.stdout) as pbar:
#         for idx_batch, batch in enumerate(training_loader, start=1):
#             x, x_len = batch.d
#
#             model.train()
#
#             y_hat = model(x)
#             y_gt = x[1:, :]  # drop <sos> in every sequence..
#             y_hat = y_hat[:(y_hat.shape[0] - 1), :, :]  # grab only y1..yk-1 from every output sequence
#             S, B, V = y_hat.shape
#
#             # CUT VERSION:
#             y_gt = y_gt.reshape(S * B)
#             y_hat = y_hat.reshape(S * B, V)
#
#
#             # calculate loss now:
#             optimizer.zero_grad()
#             loss = loss_function(y_hat, y_gt)
#             loss.backward()
#
#             # optimizing weights
#             optimizer.step()
#
#             losses.append(loss.detach())
#             pbar.update();
#             pbar.set_description(f'train loss={losses[-1]:.3f}')
#             if max_batches and idx_batch >= max_batches:
#                 break
#         epoch_list.append(i)
#         mean_loss = torch.mean(torch.FloatTensor(losses))
#         pbar.set_description(f'train loss={mean_loss:.3f}')
#
#     return [mean_loss]