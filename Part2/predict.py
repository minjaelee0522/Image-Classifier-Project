# python '/home/workspace/paind-project/predict.py' '/home/workspace/aipnd-project/flowers/test/101/image_07949.jpg'

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

parser.add_argument("input")
parser.add_argument("checkpoint")
parser.add_argument("--arch", default='vgg16')
parser.add_argument("--topk", type=int)
parser.add_argument("--gpu", default=True)
parser.add_argument("--category_names", default='/home/workspace/aipnd-project/cat_to_name.json')

parsed = parser.parse_args()

if parsed.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
else:
    model = models.densenet161(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

criterion = nn.NLLLoss()

def load_checkpoint(resume):
    if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                model.classifier = checkpoint['classifier']
                model.load_state_dict(checkpoint['state_dict'])
                model.class_to_idx = checkpoint['class_to_idx']
                
                optimizer = optim.Adam(model.classifier.parameters())
                
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))

resume = 'checkpoint.pth.tar'
load_checkpoint(resume)

if parsed.gpu == True:
    model.to('cuda')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    if im.size[0]>=im.size[1]:
        im.thumbnail((10000,256))
    else:
        im.thumbnail((256,10000))

    half_the_width = im.size[0] / 2
    half_the_height = im.size[1] / 2
    im = im.crop(
        (
            half_the_width - 112,
            half_the_height - 112,
            half_the_width + 112,
            half_the_height + 112
        )
    )

    np_image = np.array(im)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np_image / 255
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    image = torch.from_numpy(np_image)
    
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    
    image = Variable(image)
    image = image.unsqueeze(0)
    
    if parsed.gpu == True:
        output = model.forward(image.float().cuda())
    else:
        output = model.forward(image.float())
    
    probs, classes = torch.topk(torch.exp(output), topk)
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each[0].item()] for each in classes[0]]
    
    return probs, top_classes

probs, classes = predict(parsed.input, model)

if parsed.topk is not None:
    probs, classes = predict(parsed.input, model, parsed.topk)
else:
    probs, classes = predict(parsed.input, model)

with open(parsed.category_names, 'r') as f:
    cat_to_name = json.load(f)

if parsed.topk is not None:
    for i in range(0, parsed.topk):
        print(cat_to_name[classes[i]])
        print(probs[0][i].item())
else:
    print(cat_to_name[classes[0]])
    print(probs[0][0].item())