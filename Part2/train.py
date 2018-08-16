# python '/home/workspace/aipnd-project/train.py' 'home/workspace/aipnd-project/flowers'

print('ENTERED')

import torch

import torch.cuda as cuda

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models

import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim

import torch.nn.functional as F

import PIL
from PIL import Image as Image

import json

import os

import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image as Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser(description='Parse input.') 
parser.add_argument("data_dir")
parser.add_argument("--save_dir", default='/')
parser.add_argument("--arch", default='vgg16')
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--hidden_units", type=int, default=500)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gpu", default=True) # , action='store_const', const=True
parser.add_argument("--category_names", default='/home/workspace/aipnd-project/cat_to_name.json')

parsed = parser.parse_args()

data_dir = parsed.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize
                                     ])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224), # transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                      normalize
                                     ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224), # transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                      normalize
                                     ])

image_datasets = dset.ImageFolder('/home/workspace/aipnd-project/flowers/train', transform=data_transforms)
valid_datasets = dset.ImageFolder('/home/workspace/aipnd-project/flowers/valid', transform=valid_transforms)
test_datasets = dset.ImageFolder('/home/workspace/aipnd-project/flowers/test', transform=test_transforms)

dataloaders = DataLoader(image_datasets, batch_size=64, shuffle=True)
testLoaders = DataLoader(test_datasets, batch_size=64)
validLoaders = DataLoader(valid_datasets, batch_size=64)

with open(parsed.category_names, 'r') as f: # '/home/workspace/aipnd-project/cat_to_name.json'
    cat_to_name = json.load(f)

if parsed.arch == 'vgg16':
    model = models.vgg16(pretrained=True) # vgg16
else:
    model = models.densenet161(pretrained=True) # vgg16

for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(int(224*224/2) if parsed.arch == 'vgg16' else 2208, parsed.hidden_units)), # 500
                          ('relu', nn.ReLU()),
                          ('dr1', nn.Dropout(p = 0.1)),
                          ('fc2', nn.Linear(parsed.hidden_units, len(image_datasets.classes))),
                          ('dr2', nn.Dropout(p = 0.2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=parsed.learning_rate) # 0.001

epochs = parsed.epochs
print_every = 40

steps_train = 0
steps_validate = 0

if parsed.gpu == True:
    model.to('cuda')

correct = 0
total = 0

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(dataloaders):
        steps_train += 1
        
        if parsed.gpu == True:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps_train % print_every == 0:
            print("Training:")
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0
    
    for ii, (inputs, labels) in enumerate(validLoaders):
        steps_validate += 1
        
        if parsed.gpu == True:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()
        
        if steps_validate % print_every == 0:
            print("Validation:")
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0

print('Accuracy of the network on the 10000 validation images: %d %%' % (100 * correct / total))

correct = 0
total = 0
with torch.no_grad():
    for data in testLoaders:
        images, labels = data
        if parsed.gpu == True:
            images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, parsed.save_dir + filename)

# model.class_to_idx = image_datasets.classes

save_checkpoint({
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'classifier': model.classifier,
            'class_to_idx': image_datasets.class_to_idx # image_datasets.classes
        })