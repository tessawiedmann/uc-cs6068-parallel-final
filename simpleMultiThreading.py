import threading
import time
#c = threading.Condition()
import numpy as np
import time


class ComputationalThreads(threading.Thread):
    def __init__(self, name,matrix, colNum,layer):
        threading.Thread.__init__(self)
        self.name = name
        self.colNum = colNum
        self.matrix = arr
        self.maxLayerCount = 0
        self.setLayer(layer)

    def setLayer(self,layer):
        self.layer = layer
        if self.layer == 0:
            self.maxLayerCount = 150
        else:
            self.maxLayerCount = 10
    def setColumnNumber(self,colNum):
        self.colNum = colNum
    def setLayerColumnNumer(self,layer,colNum):
        self.setLayer(layer)
        self.setColumnNumber(colNum)
    def setMatrix(self,arr):
        self.matrix = arr
    def run(self):
        if self.colNum < len(self.matrix[self.layer]) :
            for i in range(len(self.matrix )):
                self.matrix[i][self.colNum] += 1


MAXNUM = 1000
arr = np.zeros((MAXNUM ,MAXNUM ))
x3 = np.random.randint(10, size=(1,150))  # Three-dimensional array
np
threadList = []
for i in range(MAXNUM):
    threadList.append(ComputationalThreads("myThread_name_A", arr, i))
start_time = time.time()
for i in range(MAXNUM):
    threadList[i].start()
print("\nsTARTEd AlL")

for i in range(MAXNUM):
    threadList[i].join()
end_time = time.time()
print("---total Time: %s seconds ---" % (end_time - start_time))
print(arr)
print("------------------------------------")