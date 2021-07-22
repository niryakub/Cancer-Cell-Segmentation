import matplotlib.pyplot as plt
import torch
import numpy as np


# calc normalization values of given data loader
def calc_normalization(data_loader):
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for idx_batch, (images, masks) in enumerate(data_loader, 0):
     # shape (batch_size, 3, height, width)
     numpy_image = images.numpy()

     # shape (3,)
     batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
     batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
     batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

     pop_mean.append(batch_mean/255)
     pop_std0.append(batch_std0/255)
     pop_std1.append(batch_std1/255)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    print(pop_mean)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    print(pop_std0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    print(pop_std1)


# VISUALIZING SAMPLES - designed for input that is the dataloader output, for raw input cancel moveaxis
def vis_sample(image, masks):
    fig, axs = plt.subplots(1,7)
    fig.tight_layout()
    axs[0].imshow(np.moveaxis(image.numpy().astype(np.uint8),0,-1)); axs[0].set_title("Sample")
    #masks = np.moveaxis(masks,0,-1)
    masks = masks.permute(1,2,0)
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
      one_hot_masks = torch.nn.functional.one_hot(torch.argmax(pred_masks, dim=0))
      one_hot_masks = one_hot_masks.permute(2, 0, 1)
      one_hot_masks *= 250  # white color?

      print("Prediction:")
      fig, axs = plt.subplots(1,7)
      fig.set_size_inches(18.5, 10.5)
      fig.tight_layout()
      axs[0].imshow(np.moveaxis(image.numpy().astype(np.uint8),0,-1)); axs[0].set_title("Sample")
      axs[1].imshow(one_hot_masks[0,:,:].cpu().detach().numpy(), cmap="gray"); axs[1].set_title("Neoplastic cells")
      axs[2].imshow(one_hot_masks[1,:,:].cpu().detach().numpy(), cmap="gray"); axs[2].set_title("Inflammatory")
      axs[3].imshow(one_hot_masks[2,:,:].cpu().detach().numpy(), cmap="gray"); axs[3].set_title("Connective/Soft tissue cells")
      axs[4].imshow(one_hot_masks[3,:,:].cpu().detach().numpy(), cmap="gray"); axs[4].set_title("Dead Cells")
      axs[5].imshow(one_hot_masks[4,:,:].cpu().detach().numpy(), cmap="gray"); axs[5].set_title("Epithelial")
      axs[6].imshow(one_hot_masks[5,:,:].cpu().detach().numpy(), cmap="gray"); axs[6].set_title("Background")
      plt.show()

      print("Ground Truth:")
      vis_sample(image,masks_gt_batch[i,...]) # visualizing gt (ground truth)