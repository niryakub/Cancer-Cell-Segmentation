from Dataset import load_dataset
from Model import UNet
from torchsummary import summary

root = 'C:\\Users\\1\\Google Drive\\PanNuke\\'
images_dir = 'Images 1\\images.npy'
masks_dir = 'Masks 1\\masks.npy'
#pannuke = PanNuke(dir_root=root, dir_masks=masks_dir, dir_images=images_dir)

# creating data loaders
training_loader, validation_loader = load_dataset(dir_root=root, dir_masks=masks_dir, dir_images=images_dir, training_size=0.8)

model = UNet()
summary(model, (3,256,256))

# # VISUALIZING SAMPLES
# rand_idx = random.randint(0,pannuke.images.shape[0])
# image = pannuke.images[rand_idx,...]
# masks = pannuke.masks[rand_idx,...]
# vis_sample(image,masks)

print("END")