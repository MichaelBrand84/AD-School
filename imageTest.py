from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

A = imread('flower.png')
X = np.mean(A, -1)
img = plt.imshow(X)



plt.show()