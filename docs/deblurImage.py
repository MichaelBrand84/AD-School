########################################################################################################
#                                                                                                      #
#	Michael Brand                                                                                      #    
#	Deblurring an image using the gradient descent method                                              #
#	PYTHON 3.8.2                                                                                       #
#                                                                                                      #
#   Based on an idea of Max Slater                                                                     #
#   https://thenumb.at/Autodiff/ [Last access 02.03.2023]                                              #
#                                                                                                      #
#   Using Code from                                                                                    #
#   https://stackoverflow.com/questions/29920114/how-to-gauss-filter-blur-a-floating-point-numpy-array #
#                                                                                                      #
########################################################################################################


## Values
## kernel_size = 4 in blur(a)
## lam = 0.01
## tol = 0.5

from matplotlib.image import imread
from floataad import float2FloatAad, getGradient

import matplotlib.pyplot as plt
import math
import numpy as np


def blur(a):
    # create kernel
    kernel_size = 4
    k1 = [np.array([math.comb(kernel_size, k) for k in range(kernel_size)])]
    kernel = np.dot(np.transpose(k1), k1)
    kernel = kernel / np.sum(kernel)
    
    # convolution
    arraylist = []
    for y in range(kernel_size):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(kernel_size):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum

def loss(x):
    # Input x is an image.
    [length, width] = np.shape(x)
    temp = blur(x)  # blur input image
    temparray = np.reshape(temp, length * width)  # reshape as a vector

    # transform to List of FloatAad-Objects
    temparray = float2FloatAad(temparray)

    y = sum((temparray - blurrarray) ** 2)
    g = getGradient(temparray, y)
    return [y.value, g]



# Read Image
# Image Source: https://www.pngall.com/toy-png/download/55843 
# Image was reshaped to 30 x 30 pixels

original = imread('bear30.jpg') / 255
# convert image to grey scale
image = 1/3 * (original[:,:,0] + original[:,:,1] + original[:,:,2])
[length, width] = np.shape(image)

# blur image
blurredimage = blur(image)
blurrarray = np.reshape(blurredimage, length*width)

# initial value for gradient descent
guessimage = np.full(shape = [length, width], fill_value=0.5)

# Gradient Descent Method
lam = 0.01
tol = 0.5

[lossval, grad] = loss(guessimage)

while lossval > tol:
    [lossval, grad] = loss(guessimage) 
    diff = np.reshape(grad, [length, width])
    guessimage = guessimage - lam * diff

# plot everything
ax = plt.subplot(3,2,1)
ax.set_title("Original")
ax.set_axis_off()
plt.imshow(image, cmap = "gray")

ax = plt.subplot(3,2,2)
ax.set_title("Blurred")
ax.set_axis_off()
plt.imshow(blurredimage, cmap = "gray")

ax = plt.subplot(3,2,3)
ax.set_title("Guess")
ax.set_axis_off()
plt.imshow(guessimage, cmap = "gray")

ax = plt.subplot(3,2,4)
ax.set_title("Blurred Guess")
ax.set_axis_off()
blurredguess = blur(guessimage)
plt.imshow(blurredguess, cmap = "gray")

ax = plt.subplot(3,2,5)
ax.set_title("Original - Guess")
ax.set_axis_off()
diffOrig = 0.5 * (image - guessimage + 1)
plt.imshow(diffOrig, cmap = "gray")

ax = plt.subplot(3,2,6)
ax.set_title("Blurred - Blurred Guess")
ax.set_axis_off()
diffBlurred = 0.5 * (blurredimage - blurredguess + 1)
plt.imshow(diffBlurred, cmap = "gray")

plt.show()