## Funktioniert nicht weil maximale Rekursionstiefe überschritten wird

import numpy as np
import matplotlib.pyplot as plt
from time import time

from floataad import FloatAad, float2FloatAad, getValues, getGradient
import mathaad


@np.vectorize
def relu(x):
    if type(x) in [int, float]:
        if x > 0:
            return x
        return 0
    else:
        if x.value > 0:
            return x
        return FloatAad(0)


def softmax(z):
    '''
    Softmax Function
    '''
    if z.dtype == "object":
        return [mathaad.exp(x) / sum(mathaad.exp(z)) for x in z]
    else:
        return [np.exp(x) / sum(np.exp(z)) for x in z]


def loss(Weights, Bias, XTrain, YTrain):
    N = len(YTrain)

    # Gewichte für Hidden- und Outputlayer als FlaotAad Matrix
    WTemp = float2FloatAad(Weights)
    WTempHidden = WTemp[:784*784]
    WtempHiddenMatrix = np.reshape(WTempHidden, [784, 784])
    WTempOutput = WTemp[784*784:]
    WtempOutputMatrix = np.reshape(WTempOutput, [784, 10])
    BiasHidden = Bias[:784]
    BiasOutput = Bias[784:]

    # Labels als One-Hot Encoding
    YOneHot = np.zeros([N, 10])
    YOneHot[range(N), YTrain.astype(int)] = 1

    # Hidden Layer
    ZHidden = XTrain @ WtempHiddenMatrix + BiasHidden
    Zhat = np.apply_along_axis(relu, 1, ZHidden)
    print("Hidden Layer Step Done")

    # Output Layer
    Zout = Zhat @ WtempOutputMatrix + BiasOutput
    Yhat = np.apply_along_axis(softmax, 1, Zout)
    D = [- YOneHot[i] @ mathaad.log(Yhat[i]) for i in range(N) ]
    J = sum(D) / N
    LossValue = getValues(J)
    LossGrad = np.array(getGradient(WTemp, J))
    return [LossValue, LossGrad]






# Trainingsdaten einlesen
f = open('fashion-mnist_train.csv', 'r')
datasetTrain = np.loadtxt(f, delimiter=',', skiprows = 1)
datasetTrain = datasetTrain / 255
f.close()
XTrain = datasetTrain[:, 1:]
YTrain = datasetTrain[:, 0]
datarecords = len(YTrain)

# Gewichte Initialisieren
W = np.random.random(784*784 + 10*784)
b = np.ones(784 + 10)

# Fit mit Stochastic Gradient Descent
lam = 0.5   # Lernrate
nEpochs = 2 # Anzahl Epochen
batchsize = 3
cycles = datarecords // batchsize
values = []

for _ in range(nEpochs):
    allData = np.arange(0, XTrain.shape[0])
    for _ in range(cycles):
        print("Cycle started")
        trainIndices = np.random.choice(XTrain.shape[0], size = batchsize, replace = False)
        allData = np.delete(allData, trainIndices)
        X = XTrain[trainIndices, :]
        Y = YTrain[trainIndices]

        # Gradient Descent Schritt 
        [Lval, Lgrad] = loss(W, b, X, Y)
        W = W - lam * Lgrad
        values.append(Lval)
    print("Epoch done")

print(values)