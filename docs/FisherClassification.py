##############################################################
#                                                            #
#	Michael Brand                                            #    
#	A Simple Neural Network for classifying Fisher Iris      #
#	PYTHON 3.8.2                                             #
#                                                            #
##############################################################

import numpy as np
import matplotlib.pyplot as plt
from time import time

from floataad import float2FloatAad, getValues, getGradient
import mathaad


def softmax(z):
    '''
    Softmax Function
    '''
    if z.dtype == "object":
        return [mathaad.exp(x) / sum(mathaad.exp(z)) for x in z]
    else:
        return [np.exp(x) / sum(np.exp(z)) for x in z]
    
def loss(Weights, bias, XTrain, YTrain):
    '''
    Cross Entropy Loss Function
    '''
    N = len(YTrain) # Anzahl Trainingsdaten
    
    # Weights as FloatAad-Matrix
    Wtemp = float2FloatAad(Weights)
    WtempMatrix = np.reshape(Wtemp, [4, 3])
        
    # Labels as One-Hot Encoding
    YOneHot = np.zeros([N, 3])
    YOneHot[range(N), YTrain] = 1

    Z = XTrain @ WtempMatrix + bias
    # apply Softmax to each row
    Yhat = np.apply_along_axis(softmax, 1, Z)
    
    # Cross Entropy
    D = [- YOneHot[i] @ mathaad.log(Yhat[i]) for i in range(N) ]
    J = sum(D) / N
    LossValue = getValues(J)
    LossGrad = np.array(getGradient(Wtemp, J))
    return [LossValue, LossGrad]

# Read Data and change labels:
fString = open('iris.data','r')
fFloat  = open('iris.csv','w')

for line in fString:
    line = line.replace('Iris-setosa', '0')
    line = line.replace('Iris-versicolor', '1')
    line = line.replace('Iris-virginica', '2')
    fFloat.write(line)

fString.close()
fFloat.close()

fFloat = open('iris.csv','r')
dataset = np.loadtxt(fFloat, delimiter = ',')
fFloat.close()

'''# plot the data
fig = plt.figure(1)

ax = fig.add_subplot(2,2,1)
ax.scatter(dataset[0:50,0], dataset[0:50,1], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,0], dataset[50:100,1], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,0], dataset[100:150,1], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kelchblattlaenge (cm)')
ax.set_ylabel('Kelchblattbreite (cm)')

ax = fig.add_subplot(2,2,2)
ax.scatter(dataset[0:50,2], dataset[0:50,3], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,2], dataset[50:100,3], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,2], dataset[100:150,3], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kronblattlaenge (cm)')
ax.set_ylabel('Kronblattbreite (cm)')

ax = fig.add_subplot(2,2,3)
ax.scatter(dataset[0:50,0], dataset[0:50,2], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,0], dataset[50:100,2], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,0], dataset[100:150,2], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kelchblattlaenge (cm)')
ax.set_ylabel('Kronblattlaenge (cm)')

ax = fig.add_subplot(2,2,4)
ax.scatter(dataset[0:50,1], dataset[0:50,3], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,1], dataset[50:100,3], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,1], dataset[100:150,3], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kelchblattbreite (cm)')
ax.set_ylabel('Kronblattbreite (cm)')

plt.show()
'''

# Split data into training and test data
X = dataset[:, 0:4] # Messwerte
Y = dataset[:, 4]   # Label
allData = np.arange(0, X.shape[0])
testIndices = np.random.choice(X.shape[0], size = 30, replace = False)
trainIndices = np.delete(allData, testIndices)
dataRecords = len(testIndices)
XTrain = X[trainIndices, :]
YTrain = np.array(Y[trainIndices], dtype = np.int32)
XTest = X[testIndices, :]
YTest = Y[testIndices]

# Initialize weights
W = np.random.random(4 * 3)
b = np.ones(3)  # bias

# Fit with Gradient Descent
lam = 0.5 # Learning Rate
tol = 1e-2

start = time()
[Lval, Lgrad] = loss(W, b, XTrain, YTrain)
while np.linalg.norm(Lgrad) > tol:
    W1 = W - lam * Lgrad
    [Lval, Lgrad] = loss(W1, b, XTrain, YTrain)
    W = W1
end = time()
elapsed = end - start
print("Der Lernprozess dauerte %1.2f Sekunden." %elapsed)

# Test the model
W = np.reshape(W, [4, 3])
Z = XTest @ W + b
Yp = Yp = np.apply_along_axis(softmax, 1, Z)
Y = np.apply_along_axis(np.argmax, 1, Yp)

# compare predictions with labels
nCorrect = sum(Y == YTest)
print("%d von %d wurden korrekt klassifiziert." %(nCorrect, dataRecords))