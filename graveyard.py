# # creating data loaders
# images_dir = 'Images 1/images.npy'
# masks_dir = 'Masks 1/masks.npy'
# training_loader_1, _ = load_dataset(dir_root='', dir_masks=masks_dir, dir_images=images_dir, training_size=1)
# images_dir = 'Images 3/images.npy'
# masks_dir = 'Masks 3/masks.npy'
# training_loader_2, _ = load_dataset(dir_root='', dir_masks=masks_dir, dir_images=images_dir, training_size=1)
# images_dir = 'Images 2/images.npy'
# masks_dir = 'Masks 2/masks.npy'
# validation_loader, test_loader = load_dataset(dir_root='', dir_masks=masks_dir, dir_images=images_dir, training_size=0.5)
#
#
#
#
# # LOADING DATASET
# def load_dataset(dir_root, dir_images, dir_masks, training_size=0.8):
#
#     train_set = PanNuke(dir_root, dir_images, dir_masks, train=True)
#
#     # Splitting train into train/val
#     permutations = torch.randperm(len(train_set))
#     split = int(np.floor(training_size * len(train_set)))
#     training_subset = SubsetRandomSampler(permutations[:split])
#     validation_subset = SubsetRandomSampler(permutations[split:])
#
#     # Apply DataLoader over train val and test data
#     train_loader = DataLoader(train_set, sampler=training_subset, batch_size=4, num_workers=4)
#     validation_loader = DataLoader(train_set, sampler=validation_subset, batch_size=4, num_workers=4)
#
#     return train_loader, validation_loader
#
#
#
# # DATASET CLASS
# class PanNuke(Dataset):
#     def __init__(self, dir_root, dir_images, dir_masks, val=False, train=False, test=False):
#         self.images = np.load(dir_root+dir_images, mmap_mode='r')
#         # self.images = np.moveaxis(self.images, -1, 1)
#         # self.images = np.moveaxis(self.images, -1, 2)
#         #self.images = np.copy(self.images)
#         self.masks = np.load(dir_root+dir_masks, mmap_mode='r')
#         self.masks = np.moveaxis(self.masks, -1, 1)
#         self.masks = np.moveaxis(self.masks, -1, 2)
#         #self.masks = np.copy(self.masks)
#         #self.masks = self.masks[:,:5,...]
#
#         self.transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
#             #transforms.Normalize(mean=[0.73952988, 0.5701334, 0.702605], std=[0.18024648, 0.21097612, 0.16465892 ])
#         ])
#
#     def __len__(self):
#         return self.images.shape[0]
#
#     def __getitem__(self, idx):
#         img = np.copy(self.images[idx, ...])
#         img = self.transforms(img)
#         masks = np.copy(self.masks[idx, ...])
#         masks = np.ceil(masks/1000)
#         return img, masks

#
# def dice_loss(predict, target):
#     smooth = 1.
#     loss = 0.
#     for c in range(predict.shape[1]):
#         iflat = predict[:, c, ...].contiguous().view(-1)
#         tflat = target[:, c, ...].contiguous().view(-1)
#         intersection = (iflat * tflat).sum()
#
#         loss +=  (1 - ((2. * intersection + smooth) /
#                           (iflat.sum() + tflat.sum() + smooth)))
#     return loss
# def BCE(predict, target):
#     loss_func = nn.CrossEntropyLoss()
#     softmax = torch.nn.Softmax(dim=0)
#     assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#     loss = 0
#     predict = predict.permute(0,2,3,1)
#     predict = predict.flatten()
#     predict = predict.reshape(5,-1).transpose()
#     predict = softmax(predict)
#     target = target.permute(0,2,3,1)
#     target = target.flatten()
#     target = target.reshape(5,-1)
#
#     loss += loss_func(predict,target)
#     return loss
#
#
# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     Args:
#         smooth: A float number to smooth loss, and avoid NaN error, default: 1
#         p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
#         predict: A tensor of shape [N, *]
#         target: A tensor of shape same with predict
#         reduction: Reduction method to apply, return mean over batch if 'mean',
#             return sum if 'sum', return a tensor of shape [N,] if 'none'
#     Returns:
#         Loss tensor according to arg reduction
#     Raise:
#         Exception if unexpected reduction
#     """
#     def __init__(self, smooth=1, p=2, reduction='sum'):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#         self.reduction = reduction
#
#     def forward(self, predict, target):
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         loss = 0
#         for b_idx in range(predict.shape[0]):
#             for m_idx in range(predict.shape[1]):
#                 predict_i = predict[b_idx,m_idx,...].contiguous().view(-1)
#                 target_i = target[b_idx,m_idx,...].contiguous().view(-1)
#
#                 num = torch.sum(torch.mul(predict_i, target_i), dim=0) + self.smooth
#                 den = torch.sum(predict_i.pow(self.p) + target_i.pow(self.p), dim=0) + self.smooth
#
#                 loss = loss +  1 - num / den
#
#         # predict = predict.contiguous().view(predict.shape[0], -1)
#         # target = target.contiguous().view(target.shape[0], -1)
#
#         # num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
#         # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
#         #
#         # loss = 1 - num / den
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise Exception('Unexpected reduction {}'.format(self.reduction))