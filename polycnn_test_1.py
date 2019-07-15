"""
Ideas:
    - experiment with different batches sizes and see what runs quicker?
    - is the Polynomial part really slow
"""

import time

# coding: utf-8

# In[3]:

# 
# Training a CNN with Polynoimal Fully connected part 
# =====================
# 
# We will train a CNN whose fully connected part can be initialized to behave like a polynomial. We will train it to classify the Cifar10 dataset.
# 
# The following steps are taken directly from the pytorch tutorial. 
# 
# 
# Training an image classifier
# ----------------------------
# 
# We will do the following steps in order:
# 
# 1. Load and normalizing the CIFAR10 training and test datasets using
#    ``torchvision``
# 2. Define a Convolutional Neural Network
# 3. Define a loss function
# 4. Train the network on the training data
# 5. Test the network on the test data
# 
# 1. Loading and normalizing CIFAR10
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Using ``torchvision``, it’s extremely easy to load CIFAR10.

import torch
import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# 
# 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Let us show some of the training images, for fun.
# 
# 

# In[6]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).
# 
# 

# In[7]:


import torch.nn as nn
import torch.nn.functional as F

# import our Polynomial Convolutional Neural Network
from polycnn import Polycnn

# instantiate a network and polynomial initialization
net = Polycnn()
net.poly_init()
print('Network is polynomial initialized\n')


# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
# 
# 

# 3.a In order to describe how polynomial approxiamtion relates to classifying images, I need to see how the convolutional neural network treats its inputs. It might be possible to view the convolutional layers as a special case of a full connected network. Essential our networks are fully connected and I think they would be far too large for a naive implementation. 

# In[8]:


# what is the size of the images
print("The size of the images used for training is:")
print(images[0].shape)
# find size of some images and the output of the first call of the forward propogation of the image


# What is the ranges of values of the images?
# I think each pixel takes values [-1,1]

# In[9]:


import numpy as np
print("maximum value is ")
print(np.amax(images[0].detach().numpy()))
print("minimum vlaue is ")
print(np.amin(images[0].detach().numpy()))


# In[10]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
# 
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
# 
# 

# In[11]:
print("BEGIN THE TRAINING")

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    oldtime = time.time()
    # the 0 in enumerate is an optional start for the index i
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            itime = time.time()
            print('[%d, %5d] loss: %.3f time: %4.4f' %
                  (epoch + 1, i + 1, running_loss / 2000, itime-oldtime))
            running_loss = 0.0
            oldtime = itime

print('Finished Training')


# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
# 
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
# 
# Okay, first step. Let us display an image from the test set to get familiar.
# 
# 

# In[33]:


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# Okay, now let us see what the neural network thinks these examples above are:
# 
# 

# In[34]:


outputs = net(images)


# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
# 
# 

# In[35]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# The results seem pretty good.
# 
# Let us look at how the network performs on the whole dataset.
# 
# 

# In[36]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
# 
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:
# 
# 

# In[37]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# Okay, so what next?
# 
# How do we run these neural networks on the GPU?
# 
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
# 
# Let's first define our device as the first visible cuda device if we have
# CUDA available:
# 
# 

# In[38]:


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

#print(device)


# The rest of this section assumes that ``device`` is a CUDA device.
# 
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
# 
# .. code:: python
# 
#     net.to(device)
# 
# 
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
# 
# .. code:: python
# 
#         inputs, labels = data[0].to(device), data[1].to(device)
# 
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is realllly small.
# 
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
# 
# **Goals achieved**:
# 
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
# 
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
# 
# Where do I go next?
# -------------------
# 
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
# 
# 
# 
