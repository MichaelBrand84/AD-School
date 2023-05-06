# Jörg Frochte: Maschinelles Lernen, 2. Auflage, Hanser (2019), S. 60

import numpy as np
import matplotlib.pyplot as plt
from time import time

from floataad import float2FloatAad, getValues, getGradient
import mathaad

# Daten einlesen
# Label in 0 (Iris-setosa), bzw. 1 (Iris-versicolor), bzw. 2 (Iris-virginica) ändern
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

'''
# Daten plotten
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


# Daten in Trainings- und Testdaten aufteilen
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


"""# Version 1: Bias wird in jedem Durchgang mitgeändert

# Matrix der Trainingsdaten mit Spalte bestehend aus 1 erweitern
XTrain = np.c_[XTrain, np.ones(len(Y) - dataRecords)]

# Gewichte initialisieren
W = np.random.random(5 * 3)

# Softmax Funktion für Vektor z
def softmax(z):
    if z.dtype == "object":
        return [mathaad.exp(x) / sum(mathaad.exp(z)) for x in z]
    else:
        return [np.exp(x) / sum(np.exp(z)) for x in z]


def loss(W, XTrain, YTrain):
    
    N = len(YTrain) # Anzahl Trainingsdaten
    
    # Gewichte als FloatAad-Matrix
    Wtemp = float2FloatAad(W)
    WtempMatrix = np.reshape(Wtemp, [5, 3])
    
    
    # Labels als one hot encoding
    YOneHot = np.zeros([N, 3])
    YOneHot[range(N), YTrain] = 1

    Z = XTrain @ WtempMatrix
    # Softmax auf jede Zeile anwenden
    Yp = np.apply_along_axis(softmax, 1, Z)
    
    # Cross Entropy
    D = [- np.transpose(YOneHot[i]) @ mathaad.log(Yp[i]) for i in range(N) ]
    S = sum(D) / N
    LossValue = getValues(S)
    LossGrad = np.array(getGradient(Wtemp, S))
    return [LossValue, LossGrad]

# Fit mit Gradient Descent
lam = 0.001 # Lernrate
tol = 4e-4
# erster Gradient Descent Schritt
[Lval, Lgrad] = loss(W, XTrain, YTrain)
W1 = W - lam * Lgrad
while np.linalg.norm(W - W1) > tol:
    W = W1
    [Lval, Lgrad] = loss(W, XTrain, YTrain)
    W1 = W - lam * Lgrad
    #print(np.linalg.norm(W - W1))

#print(Lval)


# Test des Modells
XTest = np.c_[XTest, np.ones(dataRecords)]
W = np.reshape(W1, [5, 3])
Z = XTest @ W
Yp = Yp = np.apply_along_axis(softmax, 1, Z)
Y = np.apply_along_axis(np.argmax, 1, Yp)

# Vergleich mit Resultaten
nCorrect = sum(Y == YTest)
print(str(nCorrect) + " von " + str(dataRecords) + " wurden korrekt klassifiziert.")
"""



# Softmax Funktion für Vektor z
def softmax(z):
    if z.dtype == "object":
        return [mathaad.exp(x) / sum(mathaad.exp(z)) for x in z]
    else:
        return [np.exp(x) / sum(np.exp(z)) for x in z]
    
def loss(Weights, bias, XTrain, YTrain):
    
    N = len(YTrain) # Anzahl Trainingsdaten
    
    # Gewichte als FloatAad-Matrix
    Wtemp = float2FloatAad(Weights)
    WtempMatrix = np.reshape(Wtemp, [4, 3])
    
    
    # Labels als One-Hot Encoding
    YOneHot = np.zeros([N, 3])
    YOneHot[range(N), YTrain] = 1

    Z = XTrain @ WtempMatrix + bias
    # Softmax auf jede Zeile anwenden
    Yhat = np.apply_along_axis(softmax, 1, Z)
    
    # Cross Entropy
    D = [- YOneHot[i] @ mathaad.log(Yhat[i]) for i in range(N) ]
    J = sum(D) / N
    LossValue = getValues(J)
    LossGrad = np.array(getGradient(Wtemp, J))
    return [LossValue, LossGrad]

# Gewichte initialisieren
W = np.random.random(4 * 3)  # Gewichte
b = np.ones(3)  # bias

# Fit mit Gradient Descent
lam = 0.5 # Lernrate
tol = 1e-2
start = time()
# erster Gradient Descent Schritt
[Lval, Lgrad] = loss(W, b, XTrain, YTrain)
#W1 = W - lam * Lgrad
while np.linalg.norm(Lgrad) > tol:   #np.linalg.norm(W - W1) > tol:
    W1 = W - lam * Lgrad
    [Lval, Lgrad] = loss(W1, b, XTrain, YTrain)
    W = W1
    #print(np.linalg.norm(Lgrad))
end = time()
zeit = end - start
print("Der Lernprozess dauerte %1.2f Sekunden." %zeit)

# Test des Modells
W = np.reshape(W, [4, 3])
Z = XTest @ W + b
Yp = Yp = np.apply_along_axis(softmax, 1, Z)
Y = np.apply_along_axis(np.argmax, 1, Yp)

# Vergleich mit Resultaten
nCorrect = sum(Y == YTest)
print("%d von %d wurden korrekt klassifiziert." %(nCorrect, dataRecords))