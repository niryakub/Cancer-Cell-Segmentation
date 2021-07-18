from Dataset import load_dataset
from Model import UNet
from torchsummary import summary
import torch
import torch.nn as nn

from Train import train_epoch, BinaryDiceLoss

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

root = 'C:\\Users\\1\\Google Drive\\PanNuke\\'
images_dir = 'Images 1\\images.npy'
masks_dir = 'Masks 1\\masks.npy'
#pannuke = PanNuke(dir_root=root, dir_masks=masks_dir, dir_images=images_dir)

# creating data loaders
training_loader, validation_loader = load_dataset(dir_root=root, dir_masks=masks_dir, dir_images=images_dir, training_size=0.8)

# creating model
model = UNet()
summary(model, (3,256,256))
model.double()
# training
epochs = 5
loss_function = BinaryDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps =1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

for i in range (1,epochs):
 train_epoch(training_loader=training_loader, model=model, optimizer=optimizer, loss_function=loss_function)
 scheduler.step()

# # VISUALIZING SAMPLES
# rand_idx = random.randint(0,pannuke.images.shape[0])
# image = pannuke.images[rand_idx,...]
# masks = pannuke.masks[rand_idx,...]
# vis_sample(image,masks)

print("END")