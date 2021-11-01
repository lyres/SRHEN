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
CKPT    = 40

# Directory ========= change to your own directory ========
DIR_IMG = '/data/MSCOCO_320/test_images/'
DIR_MOD = '/data/trained_models/'

# Parameter
BATCH_SZ = 100
DIS_ITER = 10
TOTAL_EP = 40

# Dataset
dataset = DataLoader(MSCOCO(DIR_IMG), batch_size=BATCH_SZ)

# Network
hnet = HNet()
hnet.eval()
hnet.to(DEVICE)
mod_file = DIR_MOD + 'model_%02d.pt' % CKPT
hnet.load_state_dict(torch.load(mod_file, map_location = torch.device('cpu')))

# MACE
def evaluate(gt_labels, pr_labels):
    # Compute mean corner error
    dist = np.power(gt_labels - pr_labels, 2)
    d1   = np.sqrt(dist[:,0] + dist[:,1])
    d2   = np.sqrt(dist[:,2] + dist[:,3])
    d3   = np.sqrt(dist[:,4] + dist[:,5])
    d4   = np.sqrt(dist[:,6] + dist[:,7])
    mce  = (d1 + d2 + d3 + d4) / 4
    mace = np.mean(mce)
    return mace

# Test
print('[Testing ' + METHOD + ' on ' + DATASET + ']')
with torch.no_grad():
    labels = []
    preds  = []
    for data in dataset:
        # Batch
        patch1 = data['patch1'].to(DEVICE)
        patch2 = data['patch2'].to(DEVICE)
        label  = data['label'].to(DEVICE)
        
        # Test step
        out   = hnet(patch1, patch2)
        label = label.cpu().numpy()
        out   = out.cpu().numpy()
        labels.append(label)
        preds.append(out)
    
    labels = np.concatenate(labels, 0)
    preds  = np.concatenate(preds, 0)
    mace = evaluate(labels, preds)
    print('\tTesting on model: %02d, MACE=%.2f' % (CKPT, mace))
