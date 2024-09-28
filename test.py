# import numpy as np
# from joblib import Parallel, delayed
#
# dim1 = 2
# dim2 = 2
# dim3 = 2
#
# x = np.zeros((dim1, dim2, dim3))
#
# def process_element(ii):
#     x[ii,:,:] = np.ones((dim2, dim3))*ii
#
# Parallel(n_jobs=-1)(delayed(process_element)(ii) for ii in range(dim1))
#
# print(x)
#
# for ii in range(dim1):
#     x[ii,:,:] = np.ones((dim2,dim3)) * ii
#
# print(x)

import numpy as np
from joblib import Parallel, delayed

dim1 = 2
dim2 = 2
dim3 = 2

x1 = np.zeros((dim1, dim2, dim3))
x2 = np.zeros((dim1, dim2, dim3))
def process_element(ii):

    y = np.ones((dim2,dim3)) * ii
    yy = np.ones((dim2, dim3)) *(ii+ii)
    return y, yy

results= Parallel(n_jobs=-1)(delayed(process_element)(ii) for ii in range(dim1))

#x = np.array(x_parallel)
#xx = np.array(xx_parallel)
#print(x)
#print(xx)
for ii in range(dim1):
    x1[ii], x2[ii] = results[ii]