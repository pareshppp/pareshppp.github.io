---
layout: post
title: Dog Breed Classification using Pytorch - Part 2
categories: [projects]
tags: [image-classification, pytorch, cnn, deep-learning]
date: 2018-10-30 13:00:00 +0530
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
if use_cuda:
    print('Using GPU.')
```

    Using GPU.


### Define Parameters

Define the parameters needed in data loader and model creation.


```python
# parameters
n_epochs = 5
num_classes = 133
num_workers = 0
batch_size = 10
learning_rate = 0.01
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

Next, we will initialize the vgg16 **pre-trained** model using the `torchvision.models.vgg16` module. We will keep the whole model unchanged except the last classifier layer, where we change the number of output nodes to number of classes.


```python
# specify model architecture 
model_transfer = torchvision.models.vgg16(pretrained=True)

# modify last layer of classifier
model_transfer.classifier[6] = nn.Linear(4096, num_classes)

print(model_transfer)
```

    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace)
        (2): Dropout(p=0.5)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace)
        (5): Dropout(p=0.5)
        (6): Linear(in_features=4096, out_features=133, bias=True)
      )
    )


### Freeze Feature Gradients
We need to freeze the gradients for the feature part of the model as we do not want to re-train the weigths for those layers. We will only train the weights for the classifier section of the model.


```python
# freeze gradients for model features
for param in model_transfer.features.parameters():
    param.require_grad = False
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


```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            
            if batch_idx % 200 == 0:
                print(f"Training Batch: {batch_idx}+/{len(loaders['train'])}")
            
        ######################    
        # validate the model #
        ######################
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            
            output = model(data)
            loss = criterion(output, target)
            
            valid_loss += loss.item() * data.size(0)
            
            if batch_idx % 200 == 0:
                print(f"Validation Batch: {batch_idx}+/{len(loaders['valid'])}")

        
        train_loss = train_loss / len(loaders['train'].dataset)
        valid_loss = valid_loss / len(loaders['valid'].dataset)
        
        # print training/validation statistics 
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}')
        
        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased from {valid_loss_min} to {valid_loss}.\nSaving Model...')
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model
```

Finally, we train the model.


```python
# train the model
if use_cuda:
    model_transfer = model_transfer.cuda()

model_transfer = train(n_epochs, loaders_transfer, model_transfer, \
                       optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
```

    Training Batch: 0+/668
    Training Batch: 200+/668
    Training Batch: 400+/668
    Training Batch: 600+/668
    Validation Batch: 0+/84
    Epoch: 1 	Training Loss: 2.233159229605498 	Validation Loss: 1.1463432044326187
    Validation loss decreased from inf to 1.1463432044326187.
    Saving Model...
    Training Batch: 0+/668
    Training Batch: 200+/668
    Training Batch: 400+/668
    Training Batch: 600+/668
    Validation Batch: 0+/84
    Epoch: 2 	Training Loss: 1.570702178994874 	Validation Loss: 0.9507174207243377
    Validation loss decreased from 1.1463432044326187 to 0.9507174207243377.
    Saving Model...
    Training Batch: 0+/668
    Training Batch: 200+/668
    Training Batch: 400+/668
    Training Batch: 600+/668
    Validation Batch: 0+/84
    Epoch: 3 	Training Loss: 1.4183635966863462 	Validation Loss: 0.9120735898167788
    Validation loss decreased from 0.9507174207243377 to 0.9120735898167788.
    Saving Model...
    Training Batch: 0+/668
    Training Batch: 200+/668
    Training Batch: 400+/668
    Training Batch: 600+/668
    Validation Batch: 0+/84
    Epoch: 4 	Training Loss: 1.3522749468014983 	Validation Loss: 0.91904990312582
    Training Batch: 0+/668
    Training Batch: 200+/668
    Training Batch: 400+/668
    Training Batch: 600+/668
    Validation Batch: 0+/84
    Epoch: 5 	Training Loss: 1.3099311252910935 	Validation Loss: 0.7952524953170451
    Validation loss decreased from 0.9120735898167788 to 0.7952524953170451.
    Saving Model...


Loading in the saved model.


```python
# load the model that got the best validation accuracy (uncomment the line below)
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
```

### Test the Model

We compare the predicted outputs with target to get the number of correct predictions and then calculate the pecentage accuracy.


```python
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    test_loss = test_loss / len(loaders['test'].dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function 
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
```

    Test Loss: 0.928593
    
    
    Test Accuracy: 73% (612/836)


### Conclusion

With only 5 epochs of training we achieved an accuracy of over 70%. The loss was still decreasing, so we may have been able to get even better performance with more training. This is a huge improvement over the ~10% accuracy we got using the model we created from scratch in [Part-1](https://pareshppp.github.io/blogs/dog-breed-classification-scratch/).

The full code for this post can found at this [link](https://github.com/pareshppp/Dog-Breed-Classification/blob/master/Dog-Breed-Classification-Transfer-Learning-VGG16.ipynb).

### Acknowledgements

This project is based on Dog-Breed-Classification project created as part of Udacity's Deep Learning Nanodegree program.

VGG16 is not the most advanced model architecture for image recognition. We can get near human level accuracy by using other model architectures such as ResNet. We will look into that in a future post.
