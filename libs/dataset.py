
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx.scipy
import cupyx.scipy.ndimage
import cupyx
import libs.binvox_rw
import random

from torchvision import transforms, utils
from PIL import Image
from scipy import ndimage



def createImg(obj3d):
    minres = min(obj3d.shape[0],obj3d.shape[1],obj3d.shape[2]) 
    
    
    proj_dir = random.randint(0,1)
    sel_axis = random.randint(0,2)
    sel_idx = random.randint(1, minres - 2)
    
    if sel_axis == 0:
        if proj_dir == 0:
            img = cp.mean(obj3d[sel_idx:,:,:], sel_axis)
        else:
            img = cp.mean(obj3d[:sel_idx,:,:], sel_axis)
    elif sel_axis == 1:
        if proj_dir == 0:
            img = cp.mean(obj3d[:,sel_idx:,:], sel_axis)
        else:
            img = cp.mean(obj3d[:,:sel_idx,:], sel_axis)
    elif sel_axis == 2:
        if proj_dir == 0:
            img = cp.mean(obj3d[:,:,sel_idx:], sel_axis)
        else:
            img = cp.mean(obj3d[:,:,:sel_idx], sel_axis)


    img = torch.from_numpy(img).float()
    img = img.expand(3,img.shape[0],img.shape[1]) #convert to rgb chanels
    
    
    trans = transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize((64,64), interpolation = Image.NEAREST),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    
    img = trans(img)
    
    
    return img

def createImgs(obj3d, img_num):
    imgs = []
    for i in range(img_num):
        imgs.append(createImg(obj3d))
    
    return torch.stack(imgs)

def dataAugmentation(sample):
    
   
    
    rotation = random.randint(0,23)
    
    if rotation == 1:
        sample = cp.rot90(sample,1,(1,2))
    elif rotation == 2:
        sample = cp.rot90(sample,2,(1,2))
    elif rotation == 3:
        sample = cp.rot90(sample,1,(2,1))
    elif rotation == 4:
        sample = cp.rot90(sample,1,(0,1))
    elif rotation == 5:
        sample = cp.rot90(sample,1,(0,1))
        sample = cp.rot90(sample,1,(1,2))
    elif rotation == 6:
        sample = cp.rot90(sample,1,(0,1))
        sample = cp.rot90(sample,2,(1,2))
    elif rotation == 7:
        sample = cp.rot90(sample,1,(0,1))
        sample = cp.rot90(sample,1,(2,1))
    elif rotation == 8:
        sample = cp.rot90(sample,1,(1,0))
    elif rotation == 9:
        sample = cp.rot90(sample,1,(1,0))
        sample = cp.rot90(sample,1,(1,2))
    elif rotation == 10:
        sample = cp.rot90(sample,1,(1,0))
        sample = cp.rot90(sample,2,(1,2))
    elif rotation == 11:
        sample = cp.rot90(sample,1,(1,0))
        sample = cp.rot90(sample,1,(2,1))
    elif rotation == 12:
        sample = cp.rot90(sample,2,(1,0))
    elif rotation == 13:
        sample = cp.rot90(sample,2,(1,0))
        sample = cp.rot90(sample,1,(1,2))
    elif rotation == 14:
        sample = cp.rot90(sample,2,(1,0))
        sample = cp.rot90(sample,2,(1,2))
    elif rotation == 15:
        sample = cp.rot90(sample,2,(1,0))
        sample = cp.rot90(sample,1,(2,1))
    elif rotation == 16:
        sample = cp.rot90(sample,1,(0,2))
    elif rotation == 17:
        sample = cp.rot90(sample,1,(0,2))
        sample = cp.rot90(sample,1,(1,2))
    elif rotation == 18:
        sample = cp.rot90(sample,1,(0,2))
        sample = cp.rot90(sample,2,(1,2))
    elif rotation == 19:
        sample = cp.rot90(sample,1,(0,2))
        sample = cp.rot90(sample,1,(2,1))
    elif rotation == 20:
        sample = cp.rot90(sample,1,(2,0))
    elif rotation == 21:
        sample = cp.rot90(sample,1,(2,0))
        sample = cp.rot90(sample,1,(1,2))
    elif rotation == 22:
        sample = cp.rot90(sample,1,(2,0))
        sample = cp.rot90(sample,2,(1,2))
    elif rotation == 23:
        sample = cp.rot90(sample,1,(2,0))
        sample = cp.rot90(sample,1,(2,1))
    
    
    
    resolution = int(sample.shape[0])
    
    strategy = random.randint(0,9)
    
    if strategy == 0:
        factor = random.uniform(1.0625, 1.25)
        sample = ndimage.zoom(sample, factor, order = 0)
        startx = random.randint(0, sample.shape[0] - resolution)
        starty = random.randint(0, sample.shape[1] - resolution)
        startz = random.randint(0, sample.shape[2] - resolution)
        sample = sample[startx:startx+resolution, starty:starty+resolution, startz:startz+resolution]
    elif strategy == 1:   
        factor = random.uniform(0.9375, 0.75)
        sample = ndimage.zoom(sample, factor, order = 0)
        padxwl = random.randint(0, resolution - sample.shape[0])
        padxwr = resolution - padxwl - sample.shape[0]
        padywl = random.randint(0, resolution - sample.shape[1])
        padywr = resolution - padywl - sample.shape[1]
        padzwl = random.randint(0, resolution - sample.shape[2])
        padzwr = resolution - padzwl - sample.shape[2]   
        sample = np.pad(sample, ((padxwl, padxwr),(padywl, padywr),(padzwl, padzwr)), mode = 'edge')
    elif strategy == 2:
        padr = int(resolution/8)
        loc = 2*padr
        startx = random.randint(0,loc)
        starty = padr
        startz = padr
        sample = np.pad(sample, ((padr,padr),(padr,padr),(padr,padr)), mode = 'edge')
        sample = sample[startx:startx+resolution, starty:starty+resolution, startz:startz+resolution]
    elif strategy == 3:
        padr = int(resolution/8)
        loc = 2*padr
        startx = padr
        starty = random.randint(0,loc)
        startz = padr
        sample = np.pad(sample, ((padr,padr),(padr,padr),(padr,padr)), mode = 'edge')
        sample = sample[startx:startx+resolution, starty:starty+resolution, startz:startz+resolution]
    elif strategy == 4:
        padr = int(resolution/8)
        loc = 2*padr
        startx = padr
        starty = padr
        startz = random.randint(0,loc)
        sample = np.pad(sample, ((padr,padr),(padr,padr),(padr,padr)), mode = 'edge')
        sample = sample[startx:startx+resolution, starty:starty+resolution, startz:startz+resolution]
        
 
    return sample

class FeatureDataset(Dataset):

    def __init__(self, list_IDs, resolution, trainflag, output_type = '3d', num_cuts = 12, data_augmentation = False):
        self.list_IDs = list_IDs
        self.resolution = resolution
        self.output_type = output_type
        self.num_cuts = num_cuts
        self.data_augmentation = data_augmentation
        self.trainflag = trainflag

    def __len__(self):
        
        return len(self.list_IDs)
   
    
    def __getitem__(self, index):
        
        idx = index
        
        ID = self.list_IDs[idx][0]
        rotation = self.list_IDs[idx][1]

        
        filename = 'data/' + str(self.resolution) + '/' + ID + '.binvox'
            
        
        with open(filename, 'rb') as f:
            sample = libs.binvox_rw.read_as_3d_array(f).data

            
        if rotation == 1:
            sample = cp.rot90(sample, 2, (0,1)).copy()  
        elif rotation == 2:
            sample = cp.rot90(sample, 1, (0,1)).copy()  
        elif rotation == 3:
            sample = cp.rot90(sample, 1, (1,0)).copy()  
        elif rotation == 4:
            sample = cp.rot90(sample, 1, (2,0)).copy()  
        elif rotation == 5:
            sample = cp.rot90(sample, 1, (0,2)).copy()  
        
        
            
        if self.data_augmentation:
            sample = dataAugmentation(sample).copy()  
            
        
        label = int(os.path.basename(filename).split('_')[0])
        
        if self.output_type == '3d':
            sample = cp.expand_dims(sample, axis=3)
            sample = torch.from_numpy(sample).float()
            sample = 2*(sample - 0.5)
        elif self.output_type == '2d_multiple':
            sample = createImgs(sample, self.num_cuts)
        elif self.output_type == '2d_single':
            sample = createImgs(sample, self.num_cuts)
            label = torch.zeros(self.num_cuts, dtype=torch.int64) + label
        

        return sample, label
    

    
def createPartition(resolution = 16, num_train = 30, num_val_test = 30):
    
    num_classes = 24
    counter = cp.zeros(num_classes)
    partition = {}
    for i in range(num_classes): 
        partition['train',i] = []
        partition['val',i] = []
        partition['test',i] = []
        
    with open(os.devnull, 'w') as devnull:
        for filename in sorted(Path('data/' + str(resolution) + '/').glob('*.binvox')):
            namelist = os.path.basename(filename).split('_')
            
            
            label = int(namelist[0])
                
            counter[label] += 1
            
            items = []
            for i in range(6):
                items.append((os.path.basename(filename).split('.')[0], i))
            
            if counter[label] % 10 < 8:
                partition['train',label] += items
            elif counter[label] % 10 == 8:
                partition['val',label] += items
            elif counter[label] % 10 == 9:
                partition['test',label] += items   
    
    ret = {}
    ret['train'] = []
    ret['val'] = []
    ret['test'] = []
    
           
    for i in range(num_classes):      
        random.shuffle(partition['train',i])  
        random.shuffle(partition['val',i])  
        random.shuffle(partition['test',i])
        
        ret['train'] += partition['train',i][0:num_train]
        ret['val'] += partition['val',i][0:num_val_test]
        ret['test'] += partition['test',i][0:num_val_test]
    
    random.shuffle(ret['train'])  
    random.shuffle(ret['val'])  
    random.shuffle(ret['test'])
        
    return ret

    









