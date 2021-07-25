import sys
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

from DataUtilities import vis_sample
from Metrics import calc_eval_metrics

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

def CE(predict, target):
    cross_entropy = nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    loss = 0
    target =  torch.argmax(target, dim=1)
    loss += cross_entropy(predict,target)
    return loss

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
        for (images, masks) in training_loader:

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

def eval_model(model,data_loader):
    DICE = []
    IOU = []
    model.eval()
    with tqdm.tqdm(total=len(data_loader), file=sys.stdout) as pbar:
        for (images, masks) in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            # calculate output
            y_hat = model(images)
            y_hat = y_hat.to(device)

            # generate one hot vectors
            one_hot_masks = torch.nn.functional.one_hot(torch.argmax(y_hat, dim=1))
            one_hot_masks = one_hot_masks.permute(0, 3, 1, 2)
            one_hot_masks = one_hot_masks.to(device)

            dice_score, iou_score = calc_eval_metrics(outputs=one_hot_masks, labels=masks)
            DICE.append(dice_score)
            IOU.append(iou_score)

            pbar.update();
            pbar.set_description(f'IOU ={IOU[-1]:.3f} ; DICE ={DICE[-1]:.3f}')
        mean_dice = torch.mean(torch.FloatTensor(DICE))
        mean_iou = torch.mean(torch.FloatTensor(IOU))
        pbar.set_description(f'IOU ={mean_iou:.3f} ; DICE ={mean_dice:.3f}')

def train_on_1_batch(model, optimizer, loss_function, images, masks,vis=False):
    losses = []
    model.train()
    images = images.to(device)
    masks = masks.to(device)
    pred_masks_batch = model(images)
    for i in range(images.shape[0]): # looping on batch_size
      image = images[i,...] # 3,256,256
      pred_masks = pred_masks_batch[i,...]
      one_hot_masks = torch.nn.functional.one_hot(torch.argmax(pred_masks, dim=0))
      one_hot_masks = one_hot_masks.permute(2,0,1)
      one_hot_masks *= 250 # white color?

      if vis is True:
        print("\n\n")
        print("\t\t\t\t\t\t\t\t SAMPLE: {}".format(str(i)))
        fig, axs = plt.subplots(1,7)
        fig.set_size_inches(18.5, 3.3)
        fig.suptitle('Prediction', fontsize=18)
        #fig.tight_layout()
        axs[0].imshow(np.moveaxis(image.cpu().detach().numpy().astype(np.uint8),0,-1)); axs[0].set_title("Sample")
        axs[1].imshow(one_hot_masks[0,:,:].cpu().detach().numpy(), cmap="gray"); axs[1].set_title("Neoplastic cells")
        axs[2].imshow(one_hot_masks[1,:,:].cpu().detach().numpy(), cmap="gray"); axs[2].set_title("Inflammatory")
        axs[3].imshow(one_hot_masks[2,:,:].cpu().detach().numpy(), cmap="gray"); axs[3].set_title("Connective/Soft tissue cells")
        axs[4].imshow(one_hot_masks[3,:,:].cpu().detach().numpy(), cmap="gray"); axs[4].set_title("Dead Cells")
        axs[5].imshow(one_hot_masks[4,:,:].cpu().detach().numpy(), cmap="gray"); axs[5].set_title("Epithelial")
        axs[6].imshow(one_hot_masks[5,:,:].cpu().detach().numpy(), cmap="gray"); axs[6].set_title("Background")
        plt.show()

        vis_sample(image,masks[i,...]) # visualizing gt (ground truth)
    #calculate output
    y_hat = model(images)

    # calculate loss now:
    optimizer.zero_grad()
    loss = loss_function(y_hat, masks)
    loss.backward()

    # optimizing weights
    optimizer.step()

    # update loss bar
    losses.append(loss.detach())
    mean_loss = torch.mean(torch.FloatTensor(losses))
    return mean_loss