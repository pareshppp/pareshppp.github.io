---
layout: post
title: Dog Breed Classification using Pytorch - Part 2
categories: [projects]
tags: [image-classification, pytorch, cnn, deep-learning]
date: 2018-10-26 13:00:00 +0530
---

## Transfer Learning with VGG16 model

### Introduction

In the previous post: [Part-1](https://pareshppp.github.io/blogs/dog-breed-classification-scratch/), we had classified the images of dog breeds using a model that we created from scratch. Using that model we predicted the dog breeds with an accuracy of around 10%. With 133 dog breeds (target classes), random selection would have given us an accuracy of less than 1%. Compared to that our simple model performed reasonably well.

But ~10% accuracy is still very low. We can use a more complex model for our problem but the more complex a model is, the more time and computing power it takes to train it. To get a high enough accuracy in our problem it would take days to train a sufficiently complex model on any personal computer.

Instead, we are going to use a method called Transfer Learning to hasten the model training process.

At a fundamental level, all images share the same basic features - Edges, Curves, Gradients, Patterns, etc. As such, we do not need to train the model to recognize these features every time. Since these features are stored in a model as weight parameters, we can re-use a pre-trained model to skip the time needed to train these weights. We only need to train the weights for the final classification layer based on our particular problem. This process is known as Transfer Learning.

In this post we are going  to use a large but simple model called VGG-16.

Let's get started.

### Import Libraries


```python
import numpy as np
from glob import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

### Check Datasets

The first step is to load-in the Images and check the total size of our dataset.

> The Dog Images Dataset can be downloaded from here: [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in this project's home directory, at the location `/dogImages`. 


```python
# load filenames for dog images
dog_files = np.array(glob(os.path.join('dogImages','*','*','*')))

# print number of images in dataset
print('There are %d total dog images.' % len(dog_files))
```

    There are 8351 total dog images.


### Check CUDA Availability

Check if GPU is available.


```python
# check if CUDA is available
use_cuda = torch.cuda.is_available()
```

### Define Parameters

Define the parameters needed in data loader and model creation.


```python
# parameters
n_epochs = 5
num_classes = 133
num_workers = 0
batch_size = 10
learning_rate = 0.05
```

### Data Loaders for the Dog Dataset

In the next step we will do the following:
1. Define Transformations that will be applied to the images using `torchvision.transforms`. Transformations are also known as Augmentation. This is a pre-processing step and it helps the model to generalize to new data much better.
2. Load the image data using `torchvision.datasets.ImageFolder` and apply the transformations.
3. Create Dataloaders using `torch.utils.data.DataLoader`.  

> **Note:**
- We have created dictionaries for all three steps that are divided into train, validation and test sets.
- The Image Resize shape and mean & standard-deviation values for Normalization module were chosen so as to replicate the VGG16 model.


```python
## TODO: Specify data loaders
trans = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

data_transfer = {
    'train': datasets.ImageFolder(os.path.join('dogImages','train'), transform=trans['train']),
    'valid': datasets.ImageFolder(os.path.join('dogImages','valid'), transform=trans['valid']),
    'test': datasets.ImageFolder(os.path.join('dogImages','test'), transform=trans['test'])
}

loaders_transfer = {
    'train': DataLoader(data_transfer['train'], batch_size=batch_size, num_workers=num_workers, shuffle=True),
    'valid': DataLoader(data_transfer['valid'], batch_size=batch_size, num_workers=num_workers, shuffle=True),
    'test': DataLoader(data_transfer['test'], batch_size=batch_size, num_workers=num_workers, shuffle=True)
}
```


```python
print(f"Size of Train DataLoader: {len(loaders_transfer['train'].dataset)}")
print(f"Size of Validation DataLoader: {len(loaders_transfer['valid'].dataset)}")
print(f"Size of Test DataLoader: {len(loaders_transfer['test'].dataset)}")
```

    Size of Train DataLoader: 6680
    Size of Validation DataLoader: 835
    Size of Test DataLoader: 836


### Model Architecture

Next, we will initialize the vgg16 **pre-trained** model using the `torchvision.models.vgg16` module.


```python
# specify model architecture 
model_transfer = torchvision.models.vgg16(pretrained=True)
```

### Specify Loss Function and Optimizer

We have chosen `CrossEntropyLoss` as our loss function and `Stochastic Gradient Descent` as our optimizer.

> **Note:**
Here we are only optimizing the weights for classifier part of the model. We will not change the weights for the features part of the model.


```python
## select loss function
criterion_transfer = nn.CrossEntropyLoss()

## select optimizer
optimizer_transfer = optim.SGD(params=model_transfer.classifier.parameters(), lr=learning_rate)
```

### Train and Validate the Model

We define a function for Training and Validation. It calculates a running train & validation loss and saves the model whenever the validation loss decreases.
