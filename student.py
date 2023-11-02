# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import scipy.ndimage
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        
        # - **conv1**: convolution layer with 6 output channels, kernel size of 3, stride of 2, padding of 1
        # - **ReLU** nonlinearity
        # - **conv2**: convolution layer with 12 output channels, kernel size of 3, stride of 2, padding of 1
        # - **ReLU** nonlinearity
        # - **conv3**: convolution layer with 24 output channels, kernel size of 3, stride of 2, padding of 1
        # - **ReLU** nonlinearity
        # - **fc**:    fully connected layer with 128 output features
        # - **ReLU** nonlinearity
        # - **cls**:   fully connected layer with 16 output features (the number of classes)
        
        #input channels are 3 cuz RGB
        self.conv1 = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        #output = 32x32 x 6
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2, padding=1)
        #output = 16x16 x 12
        self.conv3 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        #output = 8x8 x 24
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.fc = nn.Linear(8*8 * 24, 128)
        self.cls = nn.Linear(128, num_classes)


        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 8*8 * 24)
        x = self.fc(x)
        x = self.relu4(x)
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN

    #gets size of batch
    total_images = len(inputs)

    #sends input batch through network
    outputProbs = net(inputs)
    
    #get the label/index of the highest output for each input
    outputs = torch.argmax(outputProbs, dim=1, keepdim=True)
    
    
    #gets number of correct predictions
    num_correct = torch.sum(outputs == labels)
    
    #calculates loss from criterion function
    loss = criterion(outputProbs, labels.squeeze())
    running_loss = loss.item()
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    #standard backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
###               DATA AUGMENTATION
#########################################################

class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        
        #x shift
        xShift = random.randint(-self.max_shift, self.max_shift)
        #y shift
        yShift = random.randint(-self.max_shift, self.max_shift)
        
        #shifts image along width
        image = np.roll(image, xShift, axis=2)
        if xShift >= 0:
            image[:, :, :xShift] = 0
        else: 
            image[:, :, (W+xShift):] = 0
        
        #shifts image along height
        image = np.roll(image, yShift, axis=1)
        if yShift >= 0:
            image[:, :yShift, :] = 0
        else:
            image[:, (H+yShift):, :] = 0

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        
        #select a random contrast
        contrast = random.uniform(self.min_contrast, self.max_contrast)
        
        #obtains the mean intensity of the image, for each channel
        meanIntensity = np.mean(image, axis=(1,2))
        
        #applies the function output = contrast * (input) + (1-contrast) * meanIntensity
        for i in range(3):
            for j in range(H):
                for k in range(W):
                    image[i][j][k] = contrast * image[i][j][k] + (1-contrast) * meanIntensity[i]
        

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W  = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        #selects a random angle
        # angle = random.uniform(-self.max_angle, self.max_angle)
        
        # #creates the rotation matrix
        # rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        
        # #gets the center of the image
        # centerX = int((W-1)/2)
        # centerY = int((H-1)/2)
        
        # #new image of zeros
        # newImage = np.zeros((3, H, W))
        
        # #loop through each pixel
        # for i in range(W):
        #     for j in range(H):
        #         #gets the coodinates of the pixel relative to the center
        #         x = i - centerX
        #         y = j - centerY
                
        #         #gets the pixel after rotating through the formula x' = R * x
        #         xnew, ynew = np.matmul(rotationMatrix, np.array([x, y]))
                
        #         #move back to the original coordinate system
        #         inew = int(xnew + centerX)
        #         jnew = int(ynew + centerY)
                
        #         #if the pixel is inside the original image, copy the pixel over
        #         if inew >= 0 and inew < W and jnew >= 0 and jnew < H:
        #             newImage[:, j, i] = image[:, jnew, inew]
        
        # image = newImage
        
        
        #selects a random angle
        angle = random.uniform(-self.max_angle, self.max_angle)
        #rotates the image
        #reshape = false so image is tstill 64x64
        #mode = constant so that the pixels not rotated = 0
        image = scipy.ndimage.rotate(image, angle, axes=(1,2), reshape=False, mode='constant', cval=0.0)


        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        
        #random.random gives (0,1)
        
        #axis 2 = horizontal
        if random.random() <= self.p:
            image = np.flip(image, axis=2).copy()
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
###             STUDENT MODEL
#########################################################

def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer

class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN

        # TODO-BLOCK-END
        return x

#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
    
    #obtain the loss
    loss = criterion(output, label)
    
    #obtain the gradient with respect to the image
    loss.backward()
    gradient = img.grad.data
    
    #torch.sign gets the matrix of 1s and -1s
    noise = epsilon * torch.sign(gradient)
    
    #applies the noise to the image
    perturbed_image = img + noise


    # TODO-BLOCK-END

    return perturbed_image, noise

