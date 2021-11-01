import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MSCOCO(Dataset):
    def __init__(self, dir_img):
        self.dir_img = dir_img
        self.length  = len(os.listdir(dir_img))
        self.imageW  = 320
        self.imageH  = 320
        self.centerX = self.imageW // 2
        self.centerY = self.imageH // 2
        self.patchW  = 128
        self.patchH  = 128
        self.radius  = 32
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Random center position
        rxy = np.array([self.centerX, self.centerY]) + np.random.randint(-self.radius, self.radius+1, 2)
        
        # Random permutation, i.e., label
        lab = np.random.randint(-self.radius, self.radius+1, 8).astype(np.float32)
        
        # Patch corner
        x0 = rxy[0] - self.patchW // 2
        x1 = x0 + self.patchW
        y0 = rxy[1] - self.patchH // 2
        y1 = y0 + self.patchH
        c_ref = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], dtype=np.float32)
        c_new = c_ref + lab.reshape(4,2)
        
        # H3*3
        tform = cv2.getPerspectiveTransform(c_ref, c_new)
        
        # Read and warp image
        img1 = cv2.imread(self.dir_img + '%06d.jpg' % idx)
        img2 = cv2.warpPerspective(img1, tform, (self.imageW, self.imageH))
        
        # Crop patch
        patch1 = img1[int(y0):int(y1), int(x0):int(x1)]
        patch2 = img2[int(y0):int(y1), int(x0):int(x1)]
        
        # Data transformation
        patch1 = self._transform(patch1)
        patch2 = self._transform(patch2)
        sample = {'patch1': patch1, 'patch2': patch2, 'label': lab}
        return sample
    
    @staticmethod
    def _transform(img):
        # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Normalize
        img = img.astype(np.float32) / 255.0
        # ToTensor
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        return img
