import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

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
def vis_predictions(data_loader, model, model_type='unet'):
    model.eval()
    image_gt_batch, masks_gt_batch = iter(data_loader).next()

    if model_type == "micronet" :
        pred_masks_batch, _, _, _ = model(image_gt_batch) # Grabbing only PO predictions.
    else:
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

# Visualizes a random batch from a given data-loader
def visualize_segmented_ground_truth(data_loader, model, model_type):
    # Colors according to the PanNuke-dataset paper:
    colors = {"1neoplastic":(255,0,0), "2inflammatory":(0,255,0), "3connective":(0,0,255), "4dead":(255,255,0), "5epithelial":(255,102,0)}
    model.eval()

    # Grab model predictions of a random batch:
    images_gt_batch, masks_gt_batch = iter(data_loader).next()

    if model_type == "micronet":
        masks_pred_batch, _, _, _ = model(images_gt_batch)  # Grabbing only PO predictions.
    else:
        masks_pred_batch = model(images_gt_batch)

    # Turn masks maps to edges maps:
    pred_masks_edges = np.zeros((masks_gt_batch.shape))
    gt_masks_edges = np.zeros((masks_gt_batch.shape))

    for i in range(images_gt_batch.shape[0]): # looping on batch_size
        # Grab the i'th prediction masks (which hold scores) and adjust it to final prediction (one hots...)
        ith_pred = masks_pred_batch[i, ...] # CH,H,W
        ith_onehot_pred_masks = torch.nn.functional.one_hot(torch.argmax(ith_pred, dim=0)) # returns HxWxCH
        ith_onehot_pred_masks = ith_onehot_pred_masks.permute(2, 0, 1) # sets dimensions so that for example, dim=2 will be now dim=0, hence will get CH,H,W

        # Grab the i'th gt masks (already onehots)
        ith_onehot_gt_masks = masks_gt_batch[i, ...]

        # Adjust pixels' values:
        ith_onehot_pred_masks[ith_onehot_pred_masks > 0] = 255
        ith_onehot_gt_masks[ith_onehot_gt_masks > 0] = 255

        # Turn prediction masks to edges maps:
        ith_onehot_pred_masks = ith_onehot_pred_masks.cpu().numpy().astype(dtype=np.uint8)
        for j in range(ith_onehot_pred_masks.shape[0]):
            edges = cv2.Canny(ith_onehot_pred_masks[j, ...], 1, 4)   # CHECK Canny's parameters...  CAN BE IMPROVED
            pred_masks_edges[i, j, ...] = edges

        # Turn gt masks to edges maps:
        ith_onehot_gt_masks = ith_onehot_gt_masks.cpu().numpy().astype(dtype=np.uint8)
        for j in range(ith_onehot_gt_masks.shape[0]):
            edges = cv2.Canny(ith_onehot_gt_masks[j, ...], 1, 4)  # CHECK Canny's parameters...  CAN BE IMPROVED
            gt_masks_edges[i, j, ...] = edges


    # Given edges-masks-maps, paint every image in batch, according to gt_edges_masks & pred_edges_masks:
    segmented_gt_images = np.copy(images_gt_batch.cpu()) # B,CH,H,W
    segmented_pred_images = np.copy(images_gt_batch.cpu())

    # For every image in batch:
    for i in range(images_gt_batch.shape[0]):
        # For every edged-mask:
        for j, color in zip(range(gt_masks_edges.shape[0]-1), colors.keys()): #-1 since we don't paint using the "Backround mask"
            # Paint gt & pred images:
            segmented_gt_images[i, :, gt_masks_edges[i, j, ...] > 0] = colors[color]
            segmented_pred_images[i, :, pred_masks_edges[i, j, ...] > 0] = colors[color]

    # Adjust segmented images dimensions for imshow() operations:
    segmented_gt_images = np.transpose(segmented_gt_images, (0,2,3,1)) # from B,CH,H,W to B,H,W,CH (np.transpose is np version of torch.permute)
    segmented_pred_images = np.transpose(segmented_pred_images, (0,2,3,1))

    # Visualize GT vs PRED:
    fig, axarr = plt.subplots(2, segmented_gt_images.shape[0], figsize=(20, 9))
    fig.suptitle('First row= Ground Truth, Second row= Prediction', fontsize=14)
    fig.tight_layout()
    for i in range(segmented_gt_images.shape[0]):
        gt_img = segmented_gt_images[i, ...].astype(dtype=np.uint8)
        pred_img = segmented_pred_images[i, ...].astype(dtype=np.uint8)

        axarr[0, i].axis('off')
        axarr[0, i].imshow(gt_img)
        axarr[1, i].imshow(pred_img)
        axarr[1, i].axis('off')

    plt.show()







