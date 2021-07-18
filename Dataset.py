import os
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

# VISUALIZING SAMPLES
def vis_sample(image,masks):
    fig, axs = plt.subplots(1,7)
    fig.tight_layout()
    axs[0].imshow(image.astype(np.uint8)); axs[0].set_title("Sample")
    axs[1].imshow(masks[:,:,0].astype(np.uint8), cmap="gray"); axs[1].set_title("Neoplastic cells")
    axs[2].imshow(masks[:,:,1].astype(np.uint8), cmap="gray"); axs[2].set_title("Inflammatory")
    axs[3].imshow(masks[:,:,2].astype(np.uint8), cmap="gray"); axs[3].set_title("Connective/Soft tissue cells")
    axs[4].imshow(masks[:,:,3].astype(np.uint8), cmap="gray"); axs[4].set_title("Dead Cells")
    axs[5].imshow(masks[:,:,4].astype(np.uint8), cmap="gray"); axs[5].set_title("Epithelial")
    axs[6].imshow(masks[:,:,5].astype(np.uint8), cmap="gray"); axs[6].set_title("Background")
    plt.show()


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
        img = self.images[idx, ...]
        masks = self.masks[idx, ...]
        return img, masks
