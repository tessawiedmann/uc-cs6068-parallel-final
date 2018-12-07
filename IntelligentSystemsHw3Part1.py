from matplotlib.ticker import FormatStrFormatter
from random import randint
#from xlwt import Workbook
import random as rand
import matplotlib.pyplot as plt
import numpy as np
import csv
import math as math
import time


def randIndexs(size, trainNum):
    testI = []
    trainI = []
    # Training indecies
    for i in range(0, trainNum):
        newVal = randint(0, size - 1)  # Random integer for new index
        while trainI.count(newVal) > 0:  # Check if index is already used
            newVal = randint(0, size - 1)
        trainI.insert(len(trainI), newVal)  # Once unique value is found, append it
    # Testing indecies
    for i in range(0, size):  # Assign any index not in train set to test
        if trainI.count(i) == 0:
            testI.insert(len(testI), i)
    return trainI, testI


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return (1 - sigmoid(z)) * (sigmoid(z))

def dotproduct(x, y):
	final = []
	for weightColumn in y.T:
		temp = 0
		counter = 0
		for output in x:
			temp += output*weightColumn[counter]
			counter += 1
		final.append(temp)
	return np.array(final).T


################################################################################################

########## MAIN PROGRAM ##########
start_time = time.time()
########## READ DATA FILES ##########
pictures = np.empty([5000, 784])
idx = 0
with open('C:\\Users\\wiedm\\OneDrive\\Documents\\18fs\\parallel\\final\\uc-cs6068-parallel-final\\MNISTnumImages5000.txt', 'r') as f:  # read each row and save it in pictures
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        pictures[idx] = row
        idx += 1

label_file = open('C:\\Users\\wiedm\\OneDrive\\Documents\\18fs\\parallel\\final\\uc-cs6068-parallel-final\\MNISTnumLabels5000.txt', "r")  # Label file
labels = label_file.read().split('\n')  # Split file by new line to extract labels
########## READ DATA FILES ##########

########## INITIALIZATIONS ##########
LR = 0.3
alpha = 0.2
layerSizes = [784, 150, 10]  # inputSize = 784; hiddenSize = 150; outputSize = 10
trainIndx, testIndx = randIndexs(5000, 4000)  # Assign index values to use for test and train data

W = []  # [[784*150],[150*10]]
deltaW = []  # [[784*150],[150*10]]
s = []  # [[784],[150],[10]] picutre, inputs into the hidden layer, inputs into the output layer
delt = []  # [[150],[10]]
s.append(np.zeros([1, layerSizes[0]]))

for i in range(0, len(layerSizes) - 1):
    W.append(np.random.normal(0.0, math.sqrt(2 / layerSizes[i]), [layerSizes[i], layerSizes[i + 1]]))
    deltaW.append(np.zeros([layerSizes[i], layerSizes[i + 1]]))
    s.append(np.zeros([1, layerSizes[i + 1]]))
    delt.append(np.zeros([1, layerSizes[i + 1]]))

y = np.zeros([layerSizes[2], 10])
np.fill_diagonal(y, 1.0, wrap=True)
trainConfMatrix = np.zeros([layerSizes[2], 10])
testConfMatrix = np.zeros([layerSizes[2], 10])
########## INITIALIZATIONS ##########

########## TRAINING ##########
trainHR = 0.0
trHR = []
error = 0
epoch = 0
while trainHR < 0.97:
    error = 0
    epoch += 1
    # TRAIN ON EACH IMAGE IN TRAINING SET
    for i in range(0, len(trainIndx)):
        # FORWARD PROP
        s[0] = np.copy(pictures[trainIndx[i]])  # input into the first layer is the picture
        inout = np.copy(s[0])
        for layer in range(0, len(layerSizes) - 1):
            #s[layer + 1] = np.dot(inout, W[layer])  # input into the hidden layer is the dot product of the output of the previous layer and the weights
            s[layer + 1] = dotproduct(inout, W[layer])
            inout = sigmoid(s[layer + 1])  # output of the hidden layer is the sigmoid of the inputs

        # COUNT ERRORS
        tempIO = np.copy(inout)
        tempIO[int(labels[trainIndx[i]])] = 0.0  # create new array and set the correct output value to 0

        if inout[int(labels[trainIndx[i]])] <= 0.75 or max(
            tempIO) >= 0.25: error += 1  # it's a miss if the actual number is less than 0.75 or the rest are above 0.25

        # CALCULATE DELT
        delt[len(layerSizes) - 2] = sigmoidPrime(s[len(s) - 1]) * (
                    y[int(labels[trainIndx[i]])] - inout)  # delt(1:output layer) = f(s(2:output layer)) * (y-yhat)
        for layer in range(len(layerSizes) - 2, 0, -1):  # layer = 1
            #delt[layer - 1] = sigmoidPrime(s[layer]) * np.dot(delt[layer], W[layer].T)  # delt(0:hidden layer) = f'(s(1:hidden layer)) * dot(delt(1:output layer), W(output to hidden))
            delt[layer - 1] = sigmoidPrime(s[layer]) * dotproduct(delt[layer], W[layer].T)
        # CALCULATE CHANGE OF WEIGHTS
        deltaW[0] = LR * np.outer(s[0], delt[0]) + alpha * deltaW[
            0]  # deltaW(0:input to hidden) = n * outer(s(0:picture input), delt(0:hidden layer)) + momentum
        for layer in range(1, len(layerSizes) - 1):  # layer = 1
            deltaW[layer] = LR * np.outer(sigmoid(s[layer]), delt[layer]) + alpha * deltaW[
                layer]  # deltaW(hidden to output) = n * outer(s(1:hidden layer output), delt(0:output layer)) + momentum

        # CHANGE WEIGHTS
        for layer in range(0, len(layerSizes) - 1):  # layer = 0,1
            W[layer] += deltaW[layer]

    trainHR = 1 - error / len(trainIndx)
    trHR.insert(len(trHR), (1 - trainHR))
    print('epoch: ', epoch, ' Hit Rate: ', trainHR)
########## TRAINING ##########
########## TRAINING SET CONFUSION MATRIX ##########
TestError = 0
for i in range(0, len(trainIndx)):
    # FORWARD PROP
    s[0] = pictures[trainIndx[i]]  # input into the first layer is the picture
    inout = s[0]
    for layer in range(0, len(layerSizes) - 1):
        #s[layer + 1] = np.dot(inout, W[layer])  # input into the hidden layer is the dot product of the output of the previous layer and the weights
        s[layer + 1] = dotproduct(inout, W[layer])
        inout = sigmoid(s[layer + 1])  # output of the hidden layer is the sigmoid of the inputs

    # INCREMENT CONFUSION MATRIX
    trainConfMatrix[int(labels[trainIndx[i]])][np.argmax(inout)] += 1

testHR = 1 - TestError / len(testIndx)
print('Testing Hit Rate: ', testHR)
########## TRAINING SET CONFUSION MATRIX ##########

########## TESTING SET CONFUSION MATRIX ##########
TestError = 0
for i in range(0, len(testIndx)):
    # FORWARD PROP
    s[0] = pictures[testIndx[i]]  # input into the first layer is the picture
    inout = s[0]
    for layer in range(0, len(layerSizes) - 1):
        #s[layer + 1] = np.dot(inout, W[layer])  # input into the hidden layer is the dot product of the output of the previous layer and the weights
        s[layer + 1] = dotproduct(inout, W[layer])
        inout = sigmoid(s[layer + 1])  # output of the hidden layer is the sigmoid of the inputs

    # COUNT ERRORS
    if np.argmax(inout) != int(labels[testIndx[i]]): TestError += 1  # if the max's match, it's a hit

    # INCREMENT CONFUSION MATRIX
    testConfMatrix[int(labels[testIndx[i]])][np.argmax(inout)] += 1

testHR = 1 - TestError / len(testIndx)
print('Testing Hit Rate: ', testHR)
########## TESTING SET CONFUSION MATRIX ##########

plt.plot(range(1, len(trHR) + 1), trHR)
plt.title('1-Hiddent Layer Network Training Hit Rate')
plt.ylabel('Hit Rate')
plt.xlabel('Epochs')
plt.show()

pathBase = ""
with open(pathBase+"HW3-1ConfMat.csv","w") as f:
    csvWriter = csv.writer(f)#,deliminator = ",")
    csvWriter.writerow(["TRAINING CONFUSION MATRIX"])
    csvWriter.writerows(trainConfMatrix)
    csvWriter.writerow(["TESTING CONFUSION MATRIX"])
    csvWriter.writerows(testConfMatrix)

with open(pathBase+"HW3-1I-HWeightMat.csv","w") as f:
    csvWriter = csv.writer(f)  #,deliminator = ",")
    csvWriter.writerows(W[0])
with open(pathBase+"HW3-1H-OWeightMat.csv","w") as f:
    csvWriter = csv.writer(f)  #,deliminator = ",")
    csvWriter.writerows(W[1])


with open(pathBase+"HW3-1HitRates.csv","w") as f:
    csvWriter = csv.writer(f)  #,deliminator = ",")
    print(trHR)
    csvWriter.writerow(np.asarray(np.transpose(trHR)).tolist())
    csvWriter.writerow(["TEST HR",str(testHR)])

with open(pathBase + "HW3-1Indicies.csv", "w") as f:
    csvWriter = csv.writer(f)#,deliminator = ",")
    for i in range(0,max(len(trainIndx),len(testIndx))):
        temp = []
        if i < len(trainIndx):
            temp.append(trainIndx[i])
        else:
            temp.append([])
        if i < len(testIndx):
            temp.append(testIndx[i])
        else:
            temp.append([])
        if i < 15:
            print(temp)
        csvWriter.writerow(temp)
end_time = time.time()
print("---total Time: %s seconds ---" % (end_time - start_time))

"""
wb = Workbook()
sheet1 = wb.add_sheet('Conf Mat')
sheet1.write(0, 0, 'TRAINING CONFUSION MATRIX')
sheet1.write(12, 0, 'TESTING CONFUSION MATRIX')
for row in range(0, 10):
    for col in range(0, 10):
        sheet1.write(row + 1, col, trainConfMatrix[row][col])
        sheet1.write(row + 13, col, testConfMatrix[row][col])

sheet2 = wb.add_sheet('I-H Weight')
for row in range(0, layerSizes[0]):
    for col in range(0, layerSizes[1]):
        sheet2.write(row, col, W[0][row][col])

sheet3 = wb.add_sheet('H-0 Weight')
for row in range(0, layerSizes[1]):
    for col in range(0, layerSizes[2]):
        sheet3.write(row, col, W[1][row][col])

sheet4 = wb.add_sheet('Hit Rates')
for i in range(0, len(trHR)):
    sheet4.write(0, i, trHR[i])
sheet4.write(1, 0, 'TEST HR')
sheet4.write(1, 1, testHR)

sheet5 = wb.add_sheet('indecies')
for row in range(0, len(trainIndx)):
    sheet5.write(row, 0, trainIndx[row])
for row in range(0, len(testIndx)):
    sheet5.write(row, 1, testIndx[row])

wb.save('HW3P1.xls')
"""