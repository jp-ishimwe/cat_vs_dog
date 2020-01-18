import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy

import os
import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir

data_root = '/home/aims/Desktop/AMMI-AIMS-Ghana-master/AMMI-AIMS-Ghana/Exercises/data/Cat_Dog_data'
 
class DatasetLoader(Dataset):
    
    def __init__(self, data_root,transform = None):
        
        
        self.file_names = os.listdir(data_root)
        self.transform = transform
        self.data_root=data_root
              
         
    def __getitem__(self, index):
                
        if torch.is_tensor(index):
            index = index.tolist()
        
        file_name = self.file_names[index]
        
        if file_name.startswith("dog"):
            label=1
        else:
            label=0
        
        img = io.imread(os.path.join(self.data_root, file_name))
        if self.transform:
            img = self.transform(img)
            

        return (img, label)
 
    def __len__(self):
        return len(self.file_names)  
    
