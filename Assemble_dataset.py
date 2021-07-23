import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

# Variables:
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# with open(".//Final_Dataset/train_pickled_data", mode='rb') as f:  # The with just manages closing of files and etc once finished.
#     data = pickle.load(f)  # load the original pickled dataset


# Grab 3 parts of the dataset:
data_array_p1_imgs = np.uint8(np.load('PanNuke_Dataset/p1/Images/images.npy'))
data_array_p1_cancer_types = np.load('PanNuke_Dataset/p1/Images/types.npy')
data_array_p1_masks = np.uint8(np.load('PanNuke_Dataset/p1/Masks/masks.npy'))

data_array_p2_imgs = np.uint8(np.load('PanNuke_Dataset/p2/Images/images.npy'))
data_array_p2_cancer_types = np.load('PanNuke_Dataset/p2/Images/types.npy')
data_array_p2_masks = np.uint8(np.load('PanNuke_Dataset/p2/Masks/masks.npy'))

data_array_p3_imgs = np.uint8(np.load('PanNuke_Dataset/p3/Images/images.npy'))
data_array_p3_cancer_types = np.load('PanNuke_Dataset/p3/Images/types.npy')
data_array_p3_masks = np.uint8(np.load('PanNuke_Dataset/p3/Masks/masks.npy'))

# Assemble masks,labels and images onto 3 arrays:
data_array_imgs_main = np.concatenate((data_array_p1_imgs, data_array_p2_imgs,data_array_p3_imgs), axis=0)
data_array_cancer_types_main = np.concatenate((data_array_p1_cancer_types, data_array_p2_cancer_types, data_array_p3_cancer_types), axis=0)
data_array_masks_main = np.concatenate((data_array_p1_masks, data_array_p2_masks, data_array_p3_masks), axis=0)

# Create test, validation and train set:
len = data_array_imgs_main.shape[0]
rand_idxs = np.random.permutation(len) # generates random permutation-array with numbers from 0 to len

# Train:
train_imgs = data_array_imgs_main[:round(len*TRAIN_SIZE), ...]
train_masks = data_array_masks_main[:round(len*TRAIN_SIZE), ...]
train_cancer_types = data_array_cancer_types_main[:round(len*TRAIN_SIZE), ...]

# Val:
val_imgs = data_array_imgs_main[round(len*TRAIN_SIZE):round(len*(TRAIN_SIZE+VAL_SIZE)), ...]
val_masks = data_array_masks_main[round(len*TRAIN_SIZE):round(len*(TRAIN_SIZE+VAL_SIZE)), ...]
val_cancer_types = data_array_cancer_types_main[round(len*TRAIN_SIZE):round(len*(TRAIN_SIZE+VAL_SIZE)), ...]
# Test:
test_imgs = data_array_imgs_main[round(len*(TRAIN_SIZE+VAL_SIZE)):len, ...]
test_masks = data_array_masks_main[round(len*(TRAIN_SIZE+VAL_SIZE)):len, ...]
test_cancer_types = data_array_cancer_types_main[round(len*(TRAIN_SIZE+VAL_SIZE)):len, ...]

# Save the trio :
# Train:
train_data_to_pickle = {'images': train_imgs, 'cancer_types': train_cancer_types, 'masks':train_masks}
filename = './/Final_Dataset/train_pickled_data'
with open(filename, 'wb') as handle:
    print("creating: ", filename)
    pickle.dump(train_data_to_pickle, handle)

# Val
val_data_to_pickle = {'images': val_imgs, 'cancer_types': val_cancer_types, 'masks':val_masks}
filename = './/Final_Dataset/val_pickled_data'
with open(filename, 'wb') as handle:
    print("creating: ", filename)
    pickle.dump(val_data_to_pickle, handle)

# Test
test_data_to_pickle = {'images': test_imgs, 'cancer_types': test_cancer_types, 'masks':test_masks}
filename = './/Final_Dataset/test_pickled_data'
with open(filename, 'wb') as handle:
    print("creating: ", filename)
    pickle.dump(test_data_to_pickle, handle)




# Some visualizations, original image VS it's 6 masks.:

# plt.figure(1)
# img = data_array_p1_imgs[99,...]
# plt.imshow(img.astype(np.uint8))
#
#
# fig2 = plt.figure(2,constrained_layout=True)
# mask = data_array_p1_masks[99,...]
# mask0 = mask[...,0]
# mask1 = mask[...,1]
# mask2 = mask[...,2]
# mask3 = mask[...,3]
# mask4 = mask[...,4]
# mask5 = mask[...,5]
#
# spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[0, 2])
# f2_ax4 = fig2.add_subplot(spec2[1, 0])
# f2_ax5 = fig2.add_subplot(spec2[1, 1])
# f2_ax6 = fig2.add_subplot(spec2[1, 2])
#
# f2_ax1.imshow(mask0.astype(np.uint8),cmap='gray')
# f2_ax1.set_title('mask0')
# f2_ax2.imshow(mask1.astype(np.uint8),cmap='gray')
# f2_ax2.set_title('mask1')
# f2_ax3.imshow(mask2.astype(np.uint8),cmap='gray')
# f2_ax3.set_title('mask2')
# f2_ax4.imshow(mask3.astype(np.uint8),cmap='gray')
# f2_ax4.set_title('mask3')
# f2_ax5.imshow(mask4.astype(np.uint8),cmap='gray')
# f2_ax5.set_title('mask4')
# f2_ax6.imshow(mask5.astype(np.uint8),cmap='gray')
# f2_ax6.set_title('mask5')
#
# plt.show()

