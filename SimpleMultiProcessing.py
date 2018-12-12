import numpy as np
import multiprocessing as mp
import time
MAXNUM = 70
ar = np.zeros((MAXNUM ,MAXNUM ))

def callback_function(result):
    x,y,data = result
    ar[x,y] = data

def worker(num):
    data = ar[num,num]+3
    return num, num, data

def apply_async_with_callback():
    pool = mp.Pool(processes=MAXNUM )
    for i in range(MAXNUM):
        pool.apply_async(worker, args = (i, ), callback = callback_function)
    pool.close()
    pool.join()
    print("Multiprocessing done!")

if __name__ == '__main__':
    ar = np.ones((MAXNUM ,MAXNUM )) #This will be used, as local scope comes before global scope
    print(ar)
    start_time = time.time()
    apply_async_with_callback()
    end_time = time.time()
    print("---total Time: %s seconds ---" % (end_time - start_time))
    print(ar)
    print("------------------------------------")
"""from multiprocessing import Pool,get_context
import time


MAXNUM = 5
arr = [[i for i in range(MAXNUM)] for n in range(MAXNUM * 2)]

def f(col):
    with get_context("spawn").Pool() as pool:
        print("INPUT Is " + str(col))
        if col < len(arr[0]):
            for i in range(len(arr)):
                arr[i][col] += 1
        print(arr)
        print("-------------------------------------------")

if __name__ == '__main__':
    #print(f(1))
    #print(f(5))
    print(arr)
    p = Pool(MAXNUM)
    start_time = time.time()
    res = p.map(f,[i for i in range(MAXNUM)])
    end_time = time.time()
    print(res)
    print("---total Time: %s seconds ---" % (end_time - start_time))
    p.close()
    print(arr)

"""