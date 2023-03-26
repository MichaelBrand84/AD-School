# Jörg Frochte: Maschinelles Lernen, 2. Auflage, Hanser (2019), S. 60

import numpy as np
import matplotlib.pyplot as plt

# Daten einlesen
# Label in 1 (Iris-setosa), bzw. 2 (Iris-versicolor), bzw. 3 (Iris-virginica) ändern
fString = open('iris.data','r')
fFloat  = open('iris.csv','w')

for line in fString:
    line = line.replace('Iris-setosa', '1')
    line = line.replace('Iris-versicolor', '2')
    line = line.replace('Iris-virginica', '3')
    fFloat.write(line)

fString.close()
fFloat.close()

fFloat = open('iris.csv','r')
dataset = np.loadtxt(fFloat, delimiter = ',')
fFloat.close()

"""
# Daten plotten
fig = plt.figure(1)

ax = fig.add_subplot(2,2,1)
ax.scatter(dataset[0:50,0], dataset[0:50,1], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,0], dataset[50:100,1], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,0], dataset[100:150,1], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kelchblattlaenge (cm)')
ax.set_xlabel('Kelchblattbreite (cm)')

ax = fig.add_subplot(2,2,2)
ax.scatter(dataset[0:50,2], dataset[0:50,3], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,2], dataset[50:100,3], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,2], dataset[100:150,3], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kronblattlaenge (cm)')
ax.set_xlabel('Kronblattbreite (cm)')

ax = fig.add_subplot(2,2,3)
ax.scatter(dataset[0:50,0], dataset[0:50,2], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,0], dataset[50:100,2], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,0], dataset[100:150,2], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kelchblattlaenge (cm)')
ax.set_xlabel('Kronblattlaenge (cm)')

ax = fig.add_subplot(2,2,4)
ax.scatter(dataset[0:50,1], dataset[0:50,3], c = 'red', s = 20, alpha = 0.6)
ax.scatter(dataset[50:100,1], dataset[50:100,3], c = 'green', marker = '^', s = 20, alpha = 0.6)
ax.scatter(dataset[100:150,1], dataset[100:150,3], c = 'blue', marker = '*', s = 20, alpha = 0.6)
ax.set_xlabel('Kelchblattbreite (cm)')
ax.set_xlabel('Kronblattbreite (cm)')

plt.show()
"""


# Testdaten extrahieren
X = dataset[:, 1:4] # Messwerte
Y = dataset[:, 4]   # Art
allData = np.arange(0, X.shape[0])
testIndices = np.random.choice(X.shape[0], size = 30, replace = False)
trainIndices = np.delete(allData, testIndices)
dataRecords = len(testIndices)
XTrain = X[trainIndices, :]
YTrain = Y[trainIndices]
XTest = X[testIndices, :]
YTest = Y[testIndices]

