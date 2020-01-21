import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

import os
import glob
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from model import CNN
from dataset import DatasetLoader
from helper import matplotlib_imshow, images_to_probs, plot_classes_preds


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

#Available classes
classes = ['cat','dog']

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# create grid of images
img_grid = torchvision.utils.make_grid(images)
# show images
matplotlib_imshow(img_grid, one_channel=False)

#Initialize and write to tensorboard
writer = SummaryWriter('runs/experiment1')
writer.add_image('sixty_four_cats_dogs_images', img_grid)

criterion = nn.CrossEntropyLoss()
model = CNN(input_size, n_features, output_size)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#Adding the model to tensorboard
writer.add_graph(model, images)
writer.close()

#uncomment to see class prediction
#plot_classes_preds(model, images, labels)

accuracy_list = []
def train(epoch, model, perm=torch.arange(0, 784).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            # ...log the running loss
            writer.add_scalar('training loss',
                              loss / 100,
                              epoch * len(train_loader) + batch_idx)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(model, data, labels),
                              global_step=epoch * len(train_loader) + batch_idx)
            loss = 0.0


def test(model, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = data.to(device), target.to(device)

        output = model(data)
        # loss = criterion(output, target)
        test_loss += criterion(output,
                               target).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        if batch_idx % 100 == 0:
            # ...log the running loss
            writer.add_scalar('test loss',
                              test_loss / 1000,
                              epoch * len(test_loader) + batch_idx)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(model, data, labels),
                              global_step=epoch * len(test_loader) + batch_idx)
            test_loss = 0.0

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
if __name__ == "__main__":

    device = torch.device('cuda')
    model = model.to(device)
    for epoch in range(15):
        train(epoch, model)
        test(model)
