import os
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# VISUALIZING SAMPLES - designed for input that is the dataloader output, for raw input cancel moveaxis
def vis_sample(image, masks):
    fig, axs = plt.subplots(1,7)
    fig.tight_layout()
    #axs[0].imshow(np.moveaxis(image.numpy().astype(np.uint8),0,-1)); axs[0].set_title("Sample")
    masks = np.moveaxis(masks,0,-1)
    axs[1].imshow(masks[:,:,0], cmap="gray"); axs[1].set_title("Neoplastic cells")
    axs[2].imshow(masks[:,:,1], cmap="gray"); axs[2].set_title("Inflammatory")
    axs[3].imshow(masks[:,:,2], cmap="gray"); axs[3].set_title("Connective/Soft tissue cells")
    axs[4].imshow(masks[:,:,3], cmap="gray"); axs[4].set_title("Dead Cells")
    axs[5].imshow(masks[:,:,4], cmap="gray"); axs[5].set_title("Epithelial")
    axs[6].imshow(masks[:,:,5], cmap="gray"); axs[6].set_title("Background")
    plt.show()

# VISUALIZING PREDICTIONS - visualizing network output on a random sample.
def vis_predictions(data_loader, model):
    image_gt_batch, masks_gt_batch = iter(data_loader).next()
    pred_masks_batch = model(image_gt_batch)
    for i in range(image_gt_batch.shape[0]): # looping on batch_size
      print("\t\t\t SAMPLE: {}".format(str(i)))
      image = image_gt_batch[i,...] # 3,256,256
      pred_masks = pred_masks_batch[i,...] # 6,256,256

      print("Prediction:")
      fig, axs = plt.subplots(1,7)
      fig.set_size_inches(18.5, 10.5)
      fig.tight_layout()
      axs[0].imshow(np.moveaxis(image.numpy().astype(np.uint8),0,-1)); axs[0].set_title("Sample")
      axs[1].imshow(pred_masks[0,:,:].cpu().detach().numpy(), cmap="gray"); axs[1].set_title("Neoplastic cells")
      axs[2].imshow(pred_masks[1,:,:].cpu().detach().numpy(), cmap="gray"); axs[2].set_title("Inflammatory")
      axs[3].imshow(pred_masks[2,:,:].cpu().detach().numpy(), cmap="gray"); axs[3].set_title("Connective/Soft tissue cells")
      axs[4].imshow(pred_masks[3,:,:].cpu().detach().numpy(), cmap="gray"); axs[4].set_title("Dead Cells")
      axs[5].imshow(pred_masks[4,:,:].cpu().detach().numpy(), cmap="gray"); axs[5].set_title("Epithelial")
      axs[6].imshow(pred_masks[5,:,:].cpu().detach().numpy(), cmap="gray"); axs[6].set_title("Background")
      plt.show()

      print("Ground Truth:")
      vis_sample(image,masks_gt_batch[i,...]) # visualizing gt (ground truth)


# LOADING DATASET
def load_dataset(dir_root, dir_images, dir_masks, training_size=0.8):

    train_set = PanNuke(dir_root, dir_images, dir_masks, train=True)

    # Splitting train into train/val
    permutations = torch.randperm(len(train_set))
    split = int(np.floor(training_size * len(train_set)))
    training_subset = SubsetRandomSampler(permutations[:split])
    validation_subset = SubsetRandomSampler(permutations[split:])

    # Apply DataLoader over train val and test data
    train_loader = DataLoader(train_set, sampler=training_subset, batch_size=4)
    validation_loader = DataLoader(train_set, sampler=validation_subset, batch_size=4)

    return train_loader, validation_loader



# DATASET CLASS
class PanNuke(Dataset):
    def __init__(self, dir_root, dir_images, dir_masks, val=False, train=False, test=False):
        self.images = np.load(dir_root+dir_images, mmap_mode='r')
        self.images = np.moveaxis(self.images, -1, 1)
        self.masks = np.load(dir_root+dir_masks, mmap_mode='r')
        self.masks = np.moveaxis(self.masks, -1, 1)


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = np.copy(self.images[idx, ...])
        #masks = np.ceil(self.masks[idx, ...]/1000)
        masks = np.copy(self.masks[idx, ...])
        masks = np.ceil(masks/1000)
        return img, masks
