import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from Network import HNet
from Dataset import MSCOCO

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
DEVICE = torch.device('cuda:0')

# Configuration
METHOD  = 'SRHEN'
DATASET = 'MSCOCO'

# Directory ========= change to your own directory ========
DIR_IMG = '/data/MSCOCO_320/train_images/'
DIR_MOD = '/data/trained_models/'
if not os.path.exists(DIR_MOD):
    os.makedirs(DIR_MOD)

# Parameter
LR_INIT  = 1e-3
LR_STEP  = 5
LR_RATE  = 0.5
BATCH_SZ = 64
TOTAL_EP = 40
DIS_ITER = 50

# Dataset
dataset = DataLoader(MSCOCO(DIR_IMG), batch_size=BATCH_SZ, shuffle=True, num_workers=16, drop_last=True)

# Network
hnet = HNet()
hnet.train()
hnet.to(DEVICE)

# Loss function, optimizer, and learning rate scheduler
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(hnet.parameters(), LR_INIT)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP, LR_RATE)

# Train
train_loss = []
for epoch in range(TOTAL_EP):
    print('[Training ' + METHOD + ' on ' + DATASET + ']')
    count    = 0
    sum_loss = 0.0
    for data in dataset:
        # Batch
        patch1 = data['patch1'].to(DEVICE)
        patch2 = data['patch2'].to(DEVICE)
        label  = data['label'].to(DEVICE)
        
        # Train step
        optimizer.zero_grad()
        out  = hnet(patch1, patch2)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        count    += 1
        sum_loss += loss.item()
        # Dispaly
        if not count % DIS_ITER:
            lr = optimizer.param_groups[0]['lr']
            print('\tEpoch: %02d, Batch: %04d, LR: %.8f, LOSS: %.2f' % (epoch+1, count, lr, sum_loss / count))
    
    # Update learning rate
    scheduler.step()
    
    # Save model and training loss
    train_loss.append(sum_loss / count)
    print('[Saving model at epoch: ', (epoch+1), ']')
    torch.save(hnet.state_dict(), DIR_MOD + 'model_%02d.pt' % (epoch+1))
    np.savetxt(DIR_MOD + 'train_loss.txt', train_loss)
    print('===============================')
