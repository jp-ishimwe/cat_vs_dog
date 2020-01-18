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
from model import CNN
from dataset import DatasetLoader



#Fill free alter the n_features and lr initializations
output_size = 2 # class label
n_features = 6 # number of feature maps
input_size = 3*224*224 # Define the image size

data_root = './Cat_Dog_data'

transformation = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

train_d = DatasetLoader(data_root + '/train', transform=transformation)
test_d = DatasetLoader(data_root + '/test', transform=transformation)

train_loader = torch.utils.data.DataLoader(train_d, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_d, batch_size=64, shuffle=True)

accuracy_list = []

def train(epoch, model, perm=torch.arange(0, 784).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    
if __name__ == "__main__":

    model_cnn = CNN(input_size, n_features, output_size)
    optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
    
    for epoch in range(0, 1):
	train(epoch, model_cnn)
	test(model_cnn)
    
    
