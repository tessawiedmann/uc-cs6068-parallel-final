from matplotlib.ticker import FormatStrFormatter
from random import randint
#from xlwt import Workbook
import matplotlib.gridspec as gridspec
import random as rand
import matplotlib.pyplot as plt
import numpy as np
import csv
import math as math
#import xlrd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return (1 - sigmoid(z)) * (sigmoid(z))


################################################################################################

########## MAIN PROGRAM ##########

########## READ DATA FILES ##########
pictures = np.empty([5000, 784])
idx = 0
with open('MNISTnumImages5000.txt', 'r') as f:  # read each row and save it in pictures
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        pictures[idx] = row
        idx += 1

label_file = open("MNISTnumLabels5000.txt", "r")  # Label file
labels = label_file.read().split('\n')  # Split file by new line to extract labels
########## READ DATA FILES ##########

########## INITIALIZATIONS ##########
LR = 0.05
alpha = 0.2
layerSizes = [784, 150, 784]  # inputSize = 784; hiddenSize = 150; outputSize = 10

W = []  # [[784*150],[150*10]]
deltaW = []  # [[784*150],[150*10]]
s = []  # [[784],[150],[10]] picutre, inputs into the hidden layer, inputs into the output layer
delt = []  # [[150],[10]]
traindigitError = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
testdigitError = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
s.append(np.zeros([1, layerSizes[0]]))

for i in range(0, len(layerSizes) - 1):
    W.append(np.random.normal(0.0, math.sqrt(2 / layerSizes[i]), [layerSizes[i], layerSizes[i + 1]]))
    deltaW.append(np.zeros([layerSizes[i], layerSizes[i + 1]]))
    s.append(np.zeros([1, layerSizes[i + 1]]))
    delt.append(np.zeros([1, layerSizes[i + 1]]))
########## INITIALIZATIONS ##########

########## READ XLS VALUES ##########
trainIndx = []
testIndx = []
workbook = xlrd.open_workbook('HW3P1.xls')
indexSheet = workbook.sheet_by_name('indecies')
for row in range(0, 4000):
    trainIndx.insert(len(trainIndx), int(indexSheet.cell_value(row, 0)))
for row in range(0, 1000):
    testIndx.insert(len(testIndx), int(indexSheet.cell_value(row, 1)))
########## READ XLS VALUES ##########

########## TRAINING ##########
trainError = 400000
epoch = 0

while trainError > 3500:
    trainError = 0
    epoch += 1
    traindigitError = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # TRAIN ON EACH IMAGE IN TRAINING SET
    for i in range(0, len(trainIndx)):
        # FORWARD PROP
        inarray = np.copy(pictures[trainIndx[i]])
        s[0] = np.copy(pictures[trainIndx[i]])  # input into the first layer is the picture
        inout = np.copy(s[0])
        for layer in range(0, len(layerSizes) - 1):
            s[layer + 1] = np.dot(inout, W[
                layer])  # input into the hidden layer is the dot product of the output of the previous layer and the weights
            inout = sigmoid(s[layer + 1])  # output of the hidden layer is the sigmoid of the inputs

        # COUNT ERRORS
        trainError += 0.5 * sum((inarray - inout) ** 2)
        traindigitError[int(labels[trainIndx[i]])] += 0.5 * sum((inarray - inout) ** 2)

        # CALCULATE DELT
        delt[len(layerSizes) - 2] = sigmoidPrime(s[len(s) - 1]) * (
                    inarray - inout)  # delt(1:output layer) = f(s(2:output layer)) * (y-yhat)
        for layer in range(len(layerSizes) - 2, 0, -1):  # layer = 1
            delt[layer - 1] = sigmoidPrime(s[layer]) * np.dot(delt[layer], W[
                layer].T)  # delt(0:hidden layer) = f'(s(1:hidden layer)) * dot(delt(1:output layer), W(output to hidden))

        # CALCULATE CHANGE OF WEIGHTS
        deltaW[0] = LR * np.outer(s[0], delt[0]) + alpha * deltaW[
            0]  # deltaW(0:input to hidden) = n * outer(s(0:picture input), delt(0:hidden layer)) + momentum
        for layer in range(1, len(layerSizes) - 1):  # layer = 1
            deltaW[layer] = LR * np.outer(sigmoid(s[layer]), delt[layer]) + alpha * deltaW[
                layer]  # deltaW(hidden to output) = n * outer(s(1:hidden layer output), delt(0:output layer)) + momentum

        # CHANGE WEIGHTS
        for layer in range(0, len(layerSizes) - 1):  # layer = 0,1
            W[layer] += deltaW[layer]

    print('epoch: ', epoch, ' Training Error: ', trainError)
########## TRAINING ##########

########## TRAINING ##########
testError = 0

# TRAIN ON EACH IMAGE IN TRAINING SET
for i in range(0, len(testIndx)):
    # FORWARD PROP
    inarray = np.copy(pictures[testIndx[i]])
    s[0] = np.copy(pictures[testIndx[i]])  # input into the first layer is the picture
    inout = np.copy(s[0])
    for layer in range(0, len(layerSizes) - 1):
        s[layer + 1] = np.dot(inout, W[
            layer])  # input into the hidden layer is the dot product of the output of the previous layer and the weights
        inout = sigmoid(s[layer + 1])  # output of the hidden layer is the sigmoid of the inputs

    # COUNT ERRORS
    testError += 0.5 * sum(
        (inarray - inout) ** 2)  # it's a miss if the actual number is less than 0.75 or the rest are above 0.25
    testdigitError[int(labels[testIndx[i]])] += 0.5 * sum(
        (inarray - inout) ** 2)  # it's a miss if the actual number is less than 0.75 or the rest are above 0.25
print('Testing Error: ', testError)
########## TESTING ##########

weightArr = W[0].T
for neuron in range(0, 150):
    imageArray = np.zeros([28, 28])
    for i in range(0, 28):
        imageArray[i] = weightArr[neuron][28 * i:28 * i + 28]
    plt.subplot(15, 10, neuron + 1)
    plt.imshow(imageArray.T, cmap='gray', interpolation='nearest')
    plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
"""
wb = Workbook()
sheet1 = wb.add_sheet('ERRORS')
sheet1.write(0, 0, 'Training Loss')
sheet1.write(0, 1, trainError)
sheet1.write(1, 0, 'Testing Loss')
sheet1.write(1, 1, testError)
sheet1.write(2, 0, 'Digits')
sheet1.write(2, 1, 'Train')
sheet1.write(2, 2, 'Test')
for row in range(3, 13):
    sheet1.write(row, 0, row - 3)
    sheet1.write(row, 1, traindigitError[row - 3])
    sheet1.write(row, 2, testdigitError[row - 3])

sheet2 = wb.add_sheet('I-H Weight')
for row in range(0, layerSizes[0]):
    for col in range(0, layerSizes[1]):
        sheet2.write(row, col, W[0][row][col])

sheet3 = wb.add_sheet('H-0 Weight')
for row in range(0, layerSizes[2]):
    for col in range(0, layerSizes[1]):
        sheet3.write(row, col, W[1].T[row][col])

sheet4 = wb.add_sheet('indecies')
for row in range(0, len(trainIndx)):
    sheet4.write(row, 0, trainIndx[row])
for row in range(0, len(testIndx)):
    sheet4.write(row, 1, testIndx[row])

wb.save('HW3P2.xls')
"""