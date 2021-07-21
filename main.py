from Dataset import load_dataset, vis_sample, vis_predictions
from Model import UNet
from torchsummary import summary
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


from Train import train_epoch, BinaryDiceLoss, eval_loss_epoch, dice_loss

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

root = 'C:\\Users\\1\\Google Drive\\PanNuke\\'

#pannuke = PanNuke(dir_root=root, dir_masks=masks_dir, dir_images=images_dir)

# creating data loaders
images_dir = 'Images 1\\images.npy'
masks_dir = 'Masks 1\\masks.npy'
training_loader_1, _ = load_dataset(dir_root=root, dir_masks=masks_dir, dir_images=images_dir, training_size=1)
images_dir = 'Images 3\\images.npy'
masks_dir = 'Masks 3\\masks.npy'
training_loader_2, _ = load_dataset(dir_root=root, dir_masks=masks_dir, dir_images=images_dir, training_size=1)
images_dir = 'Images 2\\images.npy'
masks_dir = 'Masks 2\\masks.npy'
validation_loader, test_loader = load_dataset(dir_root=root, dir_masks=masks_dir, dir_images=images_dir, training_size=0.5)


# creating model
model = UNet()
summary(model, (3,256,256))
model.double()



# training
epochs = 5
loss_function = dice_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps =1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

for i in range (1,epochs):
 print("======================== EPOCH: {} ========================".format(str(i)))
 print("TRAIN LOSS:")
 train_epoch(training_loader=training_loader_1, model=model, optimizer=optimizer, loss_function=loss_function)
 train_epoch(training_loader=training_loader_2, model=model, optimizer=optimizer, loss_function=loss_function)
 print("VAL LOSS:")
 eval_loss_epoch(training_loader=validation_loader, model=model, loss_function=loss_function)
 scheduler.step()



print("END")