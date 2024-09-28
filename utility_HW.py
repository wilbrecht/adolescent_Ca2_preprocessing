# general utility function needed in data analysis
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def bootstrap(data, dim, dim0, n_sample=1000):
    """
    input:
    data: data matrix for bootstrap
    dim: the dimension for bootstrap, should be data.shape[1]
    dim0: the dimension untouched, shoud be data.shape[0]
    n_sample: number of samples for bootstrap. default: 1000
    output:
    bootRes={'bootAve','bootHigh','bootLow'}
    """
    # Resample the rows of the matrix with replacement
    if len(data)>0:  # if input data is not empty
        bootstrap_indices = np.random.choice(data.shape[dim], size=(n_sample, data.shape[dim]), replace=True)

        # Bootstrap the matrix along the chosen dimension
        bootstrapped_matrix = np.take(data, bootstrap_indices, axis=dim)

        meanBoot = np.nanmean(bootstrapped_matrix,2)
        bootAve = np.nanmean(bootstrapped_matrix, axis=(1, 2))
        bootHigh = np.nanpercentile(meanBoot, 97.5, axis=1)
        bootLow = np.nanpercentile(meanBoot, 2.5, axis=1)

    else:  # return nans
        bootAve = np.full(dim0, np.nan)
        bootLow = np.full(dim0, np.nan)
        bootHigh = np.full(dim0, np.nan)
        # bootstrapped_matrix = np.array([np.nan])

    # bootstrapped_2d = bootstrapped_matrix.reshape(80,-1)
    # need to find a way to output raw bootstrap results
    tempData = {'bootAve': bootAve, 'bootHigh': bootHigh, 'bootLow': bootLow}
    index = np.arange(len(bootAve))
    bootRes = pd.DataFrame(tempData, index)

    return bootRes

def count_consecutive(listx):
    # count largest number of consecutive 1s in a given list
    count1 = 0
    maxConsec1 = 0
    for ii in range(len(listx)):
        if listx[ii] == 1:
            count1 = count1+1
            if count1 > maxConsec1:
                maxConsec1 = count1
        else:
            count1 = 0

    return maxConsec1

if __name__ == "__main__":
    x = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0,1, 1,1,1,1,0,1]
    print(count_consecutive(x))