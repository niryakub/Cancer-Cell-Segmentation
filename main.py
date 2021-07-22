from Dataset import load_dataset
from Model import UNet
from torchsummary import summary
import torch
from TrainingUtilities import train_epoch, CE, eval_loss_epoch, train_on_1_batch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

train_dir = 'C:\\Users\\1\\Google Drive\\PanNuke\\train_pickled_data'
val_dir = 'C:\\Users\\1\\Google Drive\\PanNuke\\val_pickled_data'
test_dir = 'C:\\Users\\1\\Google Drive\\PanNuke\\test_pickled_data'
# Grab loaders:
training_loader = load_dataset(2, shuffle_flag=True, num_workers=0, data_dir=train_dir)
val_loader = load_dataset(2, shuffle_flag=True, num_workers=0, data_dir=val_dir)
validation_loader = load_dataset(2, shuffle_flag=True, num_workers=0, data_dir=test_dir)

# creating model
model = UNet()
summary(model, (3,256,256))

iter(training_loader).next()

# training
epochs = 5
loss_function = CE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps =1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
imgs, masks = iter(training_loader).next()
for i in range (1,epochs):
    print("======================== EPOCH: {} ========================".format(str(i)))
    print("TRAIN LOSS:")
    #train_epoch(training_loader=training_loader, model=model, optimizer=optimizer, loss_function=loss_function)
    train_on_1_batch(model, optimizer, loss_function, images=imgs, masks=masks, vis=False)
    print("VAL LOSS:")
    #eval_loss_epoch(training_loader=validation_loader, model=model, loss_function=loss_function)
    #scheduler.step()

print("END")