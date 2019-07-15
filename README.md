# polycnn
Contains experiments related polynomial initialization for convolutional neural networks for image classification.

Dependencies:
- Pytorch
- Python
- Numpy

## The basic idea
Many popular architectures for image classification use a series of convulational layers to transform the input 
images into a set of features. This part of the network can be viewed as a mapping from image space to features space.
In order to perform classiciation the network then maps the features vector into class space which is typically viewed
as a vector of k probabilities, where k is the number of classes. Futhermore, many popular architecutres use
fully connected layers for the mapping from features space to class space. We propose to replace the fully connected layers
with our our fully connected networks which can be initialized to behave like polynomials. 

There will have to be an ablation analysis wherein we consider
- training the full network with full connected layers replaced by our own networks
- training only the fully connected layers while using the pretrained feature extractors of an existing network
- Currently quadratic polynomials are easily initialized, however in the future we will want to consider 
  initializing to higher degree polynomials.
- The fully connected layers are often the same width as the feature vector itself. Our networks require
  a width of at least 4d.
  
