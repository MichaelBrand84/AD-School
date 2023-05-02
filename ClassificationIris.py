# Jörg Frochte: Maschinelles Lernen, 2. Auflage, Hanser (2019), S. 60

import numpy as np
import matplotlib.pyplot as plt

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

# Matrix der Trainingsdaten mit Spalte bestehend aus 1 erweitern
XTrain = np.c_[XTrain, np.ones(len(Y) - dataRecords)]



# Gewichte initialisieren
W = np.random.random(5 * 3)




def loss(W, XTrain, YTrain):
    def softmax(z):
        return [mathaad.exp(x) / sum(mathaad.exp(z)) for x in z]
    
    N = len(YTrain) # Anzahl Trainingsdaten
    
    # Gewichte als FloatAad-Matrix
    Wtemp = float2FloatAad(W)
    Wtemp = np.reshape(Wtemp, [5, 3])
    
    
    # Labels als one hot encoding
    YOneHot = np.zeros([N, 3])
    YOneHot[range(N), YTrain] = 1

    Z = XTrain @ Wtemp
    # Softmax auf jede Zeile anwenden
    Yp = np.apply_along_axis(softmax, 1, Z)
    
    # Cross Entropy
    D = [- YOneHot[i] @ mathaad.log(Yp[i]) for i in range(N) ]
    S = sum(D) / N
    LossValue = getValues(S)
    LossGrad = getGradient(W, S)
    return [LossValue, LossGrad]


[Lval, Lgrad] = loss(W, XTrain, YTrain)
print(Lval)
print(Lgrad)  # Problem: alle Gradienten sind Null
