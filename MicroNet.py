import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TNF
import torch.nn.modules
from torch import optim
import time
import dataset as pnk
from torch.utils.data import DataLoader
from torchsummary import summary
import tqdm

# Small useful components:
# this class is to be able to use TNF.interpole within nn.Sequential()
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = TNF.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        return self.interp(x, size=self.size, mode=self.mode, align_corners=False)

# Create the Micro-Net components:
class Group1_B1(nn.Module):
    def __init__(self, in_channels=3):
        super(Group1_B1, self).__init__()

        self.sub_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False), # kernel=5 since we start with 256 imgs, where in paper it's 252
            nn.BatchNorm2d(64), # out_channels=64
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),  # REMOVE?
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        ) #124,124,64

        self.sub_block2 = nn.Sequential(
            Interpolate(size=(128,128),mode='bicubic'),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),  # out_channels=64
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, orig_input):
        sub_block1 = self.sub_block1(orig_input) # recall it outputs: B,CH,HEIGHT,WIDTH
        sub_block2 = self.sub_block2(orig_input)
        B1 = torch.cat((sub_block1,sub_block2), dim=1) # concat alongside channels dim'
        return B1

class Group1_B2(nn.Module):
    def __init__(self, in_channels=128):
        super(Group1_B2, self).__init__()
        self.sub_block1 = nn.Sequential(  # gets 124^2, ch=128,   outputs: 60^2, ch=128
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True), # bias=True since no BN is applied.
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.sub_block2 = nn.Sequential( # gets 256^2, ch=3  outputs: 60^2, ch=128
            Interpolate(size=(64, 64), mode='bicubic'),
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, B1_input, orig_input):
        sub_block1 = self.sub_block1(B1_input)
        sub_block2 = self.sub_block2(orig_input)
        B2 = torch.cat((sub_block1, sub_block2), dim=1)  # concat alongside channels dim'
        return B2

class Group1_B3(nn.Module):
    def __init__(self, in_channels=256):
        super(Group1_B3, self).__init__()
        self.sub_block1 = nn.Sequential(  # gets 60^2, ch=256,   outputs: 28^2, ch=256
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True), # bias=True since no BN is applied.
            nn.Tanh(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.sub_block2 = nn.Sequential( # gets 256^2, ch=3  outputs: 28^2, ch=256
            Interpolate(size=(32, 32), mode='bicubic'),
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, B2_input, orig_input):
        sub_block1 = self.sub_block1(B2_input)
        sub_block2 = self.sub_block2(orig_input)
        B3 = torch.cat((sub_block1, sub_block2), dim=1)  # concat alongside channels dim'
        return B3

class Group1_B4(nn.Module):
    def __init__(self, in_channels=512):
        super(Group1_B4, self).__init__()
        self.sub_block1 = nn.Sequential(  # gets 28^2, ch=512,   outputs: 12^2, ch=512
            nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True), # bias=True since no BN is applied.
            nn.Tanh(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.sub_block2 = nn.Sequential( # gets 256^2, ch=3  outputs: 12^2, ch=512
            Interpolate(size=(16, 16), mode='bicubic'),
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, B3_input, orig_input):
        sub_block1 = self.sub_block1(B3_input)
        sub_block2 = self.sub_block2(orig_input)
        B4 = torch.cat((sub_block1, sub_block2), dim=1)  # concat alongside channels dim'
        return B4

class Group2_B5(nn.Module):
    def __init__(self, in_channels=1024):
        super(Group2_B5, self).__init__()
        self.sub_block = nn.Sequential(  # gets 12^2, ch=1024,   outputs: 8^2, ch=2048
            nn.Conv2d(in_channels=in_channels, out_channels=2048, kernel_size=3, stride=1, padding=0, bias=True), # bias=True since no BN is applied.
            nn.Tanh(),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=0, bias=True),
            nn.Tanh(),
        )

    def forward(self, B4_input):
        B5 = self.sub_block(B4_input)
        return B5

class Group3_Bi(nn.Module):
    def __init__(self, in_channels_prev_b, in_channels_g1):
        super(Group3_Bi, self).__init__()
        # ------------------- UNSURE HERE REGARDING THE 1ST SUBBLOCK, WE DECONV UP TO 16 THEN CONV TWICE TO 8 THEN AGAIN DECONV UP TO 16?????
        # First-part of the block:
        self.sub_block1 = nn.Sequential(  # gets size^2, #channels, outputs (size*2)^2, #channels/2
            nn.ConvTranspose2d(in_channels=in_channels_prev_b, out_channels=round(in_channels_prev_b / 2),kernel_size=2, stride=2, padding=0), # double x=h,w to 2x X 2x
            nn.Conv2d(in_channels=round(in_channels_prev_b / 2), out_channels=round(in_channels_prev_b / 2), kernel_size=3, stride=1, padding=0), # turn to 2x-2 X 2x-2
            # nn.BatchNorm2d(in_channels_prev_b/2),
            nn.Tanh(),
            nn.Conv2d(in_channels=round(in_channels_prev_b / 2), out_channels=round(in_channels_prev_b / 2), kernel_size=3, stride=1, padding=0),  # turn to 2x -4 X 2x-4
            # nn.BatchNorm2d(in_channels_prev_b/2),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=round(in_channels_prev_b / 2), out_channels=round(in_channels_prev_b / 2), kernel_size=5, stride=1, padding=0),  # turn back to 2x X 2x,
        )

        # Mid-part of the block:
        self.sub_block2 = nn.ConvTranspose2d(in_channels=in_channels_g1, out_channels=in_channels_g1, kernel_size=5, stride=1, padding=0) # it upsamples by 4 only

        # Third-part of the block:
        self.sub_block3 = nn.Sequential(
            nn.Conv2d(in_channels=round(in_channels_g1*2), out_channels=in_channels_g1, kernel_size=3, stride=1, padding=1), # same conv
            nn.Tanh()
        )

    def forward(self, g1_input, prev_b_input):
        sub_block1 = self.sub_block1(prev_b_input)
        sub_block2 = self.sub_block2(g1_input)
        sub_block3 = torch.cat((sub_block1, sub_block2), dim=1)  # concat alongside channels dim'
        Bi = self.sub_block3(sub_block3)
        return Bi

class Group4_Pa1(nn.Module):
    def __init__(self):
        super(Group4_Pa1, self).__init__()
        self.sub_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2), #upsample by 2x
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),  #UNLIKE THE PAPER, we'll use same convs, in order to get final-output= 256x256 as out data imgs
            nn.Tanh(),
        )

        self.sub_block2 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1, padding=0), #unlike paper ^^
            #nn.Tanh(inplace=True), # Since it's output layer ^^..
        )

    def forward(self, b9_input):
        x1 = self.sub_block1(b9_input) # this also goes onwards to Group5
        pa1 = self.sub_block2(x1)
        return pa1, x1

class Group4_Pa2(nn.Module):
    def __init__(self):
        super(Group4_Pa2, self).__init__()
        self.sub_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4), #upsample by 4x
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),  #UNLIKE THE PAPER, we'll use same convs, in order to get final-output= 256x256 as out data imgs
            nn.Tanh(),
        )

        self.sub_block2 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=128, out_channels=6, kernel_size=1, stride=1, padding=0), #unlike paper ^^
            #nn.Tanh(inplace=True), # Since it's output layer ^^..
        )

    def forward(self, b8_input):
        x2 = self.sub_block1(b8_input) # this also goes onwards to Group5
        pa2 = self.sub_block2(x2)
        return pa2, x2

class Group4_Pa3(nn.Module):
    def __init__(self):
        super(Group4_Pa3, self).__init__()
        self.sub_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=8), #upsample by 8x
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),  #UNLIKE THE PAPER, we'll use same convs, in order to get final-output= 256x256 as out data imgs
            nn.Tanh(),
        )

        self.sub_block2 = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=256, out_channels=6, kernel_size=1, stride=1, padding=0), #unlike paper ^^
            #nn.Tanh(inplace=True), # Since it's output layer ^^..
        )

    def forward(self, b7_input):
        x3 = self.sub_block1(b7_input) # this also goes onwards to Group5
        pa3 = self.sub_block2(x3)
        return pa3, x3

class Group5(nn.Module):
    def __init__(self):
        super(Group5, self).__init__()
        self.sub_block = nn.Sequential( #unlike the paper, the input for this stage is 256x256, 448(after concat)
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=448, out_channels=6, kernel_size=3, stride=1, padding=1), #unlike paper ^^
        )

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1) # concat x1,x2,x3 alongside channels dim'
        p0 = self.sub_block(x)
        return p0

class MicroNet(nn.Module):
    def __init__(self):
        super(MicroNet, self).__init__()
        self.group1_b1 = Group1_B1()
        self.group1_b2 = Group1_B2()
        self.group1_b3 = Group1_B3()
        self.group1_b4 = Group1_B4()
        self.group2 = Group2_B5()
        self.group3_b6 = Group3_Bi(in_channels_prev_b=2048, in_channels_g1=1024)
        self.group3_b7 = Group3_Bi(in_channels_prev_b=1024, in_channels_g1=512)
        self.group3_b8 = Group3_Bi(in_channels_prev_b=512, in_channels_g1=256)
        self.group3_b9 = Group3_Bi(in_channels_prev_b=256, in_channels_g1=128)
        self.group4_pa1 = Group4_Pa1()
        self.group4_pa2 = Group4_Pa2()
        self.group4_pa3 = Group4_Pa3()
        self.group5 = Group5()

    def forward(self, x):
        # Propagate through G1:
        b1 = self.group1_b1(x)
        b2 = self.group1_b2(b1, x)
        b3 = self.group1_b3(b2, x)
        b4 = self.group1_b4(b3, x)

        # Propagate through G2:
        b5 = self.group2(b4)

        # Propagate through G3:
        b6 = self.group3_b6(b4, b5)
        b7 = self.group3_b7(b3, b6)
        b8 = self.group3_b8(b2, b7)
        b9 = self.group3_b9(b1, b8)

        # Propagate through G4:
        pa1, x1 = self.group4_pa1(b9)
        pa2, x2 = self.group4_pa2(b8)
        pa3, x3 = self.group4_pa3(b7)

        # Propagate through G5:
        p0 = self.group5(x1, x2, x3)

        return p0, pa1, pa2, pa3  # recall that p0 = main output, pa1,pa2,pa3 = auxiliary outputs

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def load_dataset(batch_size, shuffle_flag, num_workers, data_dir, transforms=None):
    dataset = pnk.PanNukeDataset(data_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag)#, num_workers=num_workers)
    return data_loader

def train(model, loader, opt, criterion,scheduler, epoch):
    epoch_loss = 0
    epoch_acc = 0

    # Train the model (turn training-mode on..)
    model.train()

    with tqdm.tqdm(total=len(loader), file=sys.stdout) as pbar:
        for (images, masks) in loader:
            images = images.to(device)
            masks = masks.to(device)

            # reinitialize gradients..
            opt.zero_grad()

            # Calculation of loss (according to paper):
            p0, pa1, pa2, pa3 = model(images) # Pi.shape, 256x256x5
            masks = torch.argmax(masks, dim=1)
            #loss = criterion(output, masks)
            l0 = criterion(p0, masks)
            l1 = criterion(pa1, masks)
            l2 = criterion(pa2, masks)
            l3 = criterion(pa3, masks)
            loss = l0+(l1+l2+l3)/epoch
            # Backpropagation
            loss.backward()

            # Calculate accuracy
            #acc = calculate_accuracy(output, labels)

            # update weights according to gradients
            opt.step()

            epoch_loss += loss.item()
            #epoch_acc += acc.item

            pbar.update()
            pbar.set_description(f'train loss={loss.item():.3f}')

        pbar.set_description(f'train loss={epoch_loss / len(loader):.3f}')
    scheduler.step()

    return epoch_loss / len(loader), epoch_acc / len(loader)


# ================= MAIN =====================
# Variables:
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE=0.001

train_dir = './/Final_Dataset/train_pickled_data'
val_dir = './/Final_Dataset/val_pickled_data'
test_dir = './/Final_Dataset/test_pickled_data'

# Set up device:
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

# Grab loaders:
train_loader = load_dataset(BATCH_SIZE, shuffle_flag=True, num_workers=NUM_WORKERS, data_dir=train_dir)
val_loader = load_dataset(BATCH_SIZE, shuffle_flag=True, num_workers=NUM_WORKERS, data_dir=val_dir)
test_loader = load_dataset(BATCH_SIZE, shuffle_flag=True, num_workers=NUM_WORKERS, data_dir=test_dir)

model = MicroNet().to(device)
summary(model, (3, 256, 256))

# Define optimizer and criterion functions   - IMPROVE... LEARN HP'S
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1, verbose=False)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print("Epoch-%d: " % (epoch))

    train_loss, train_acc = train(model, train_loader, optimizer, loss_func, scheduler, epoch+1)


    print(f"train loss={train_loss}, epoch={epoch}")

    # val_loss, val_acc = evaluate(model, val_loader, loss_func)

    # print(f"train loss={train_loss}, epoch={epoch}")
