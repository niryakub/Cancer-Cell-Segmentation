from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class PanNukeDataset(Dataset):
    def  __init__(self, image_dir, masks_dir=None, cancer_types_dir=None, transform=None):
        #self.image_dir = image_dir
        #self.masks_dir = masks_dir
        #self.cancer_types_dir = cancer_types_dir
        self.transform = transform

        # Data is pickled into dictionaries ({'images': imgs, 'cancer_types': cancer_types, 'masks':masks}), let's open it up:
        with open(image_dir, mode='rb') as f:  # The with just manages closing of files and etc once finished.
            data_dict = pickle.load(f)  # load the original pickled dataset

        self.images = data_dict['images']
        self.cancer_types = data_dict['cancer_types']
        self.masks = data_dict['masks']

    def __len__(self):
        return self.images.shape[0]

    # returns images,masks   FOR NOW IT DOESN'T RETURN cancer_types!!!
    def __getitem__(self, index):
        image = self.images[index, ...].astype(dtype=np.float32)
        mask = self.masks[index, ...].astype(dtype=np.float32)
        mask[mask > 0] = 1.0

        # Set dimensions from 256x256xCH, to CHx256x256  - CHECK TO BE POSITIVE
        mask = np.moveaxis(mask, -1, 0)
        image = np.moveaxis(image, -1, 0)


        if self.transform is not None :
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        return torch.from_numpy(image),torch.from_numpy(mask)

# Creating data loaders:
def load_dataset(batch_size, shuffle_flag, num_workers=None, data_dir=None, transforms=None):
  dataset = PanNukeDataset(data_dir)
  if num_workers is not None :
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=num_workers)
  else :
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)

  return data_loader
