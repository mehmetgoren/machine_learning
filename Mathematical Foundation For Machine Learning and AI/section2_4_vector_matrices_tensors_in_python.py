# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:47:47 2018

@author: mehme
"""

import sys
import numpy as np

#print(sys.version)
#print(np.__version__)

def output(A):
    print(A)
    print()
    print("Shape:", A.shape)
    print("Size: ", A.size)


#defining a vector

v = np.array([1,2,3])

print(v.shape) #dimension also
print(v.size)


#matrix
A = np.matrix([[1,2,3], [4,5,6], [7,8,9]])
print(A)
print(A.shape)
print(A.size)

A = np.zeros([10,10])#or np.ones
print(A)


print("-------------------------------------------")
print("\n\n\n")

#lets define a tensor
tensor = np.zeros([3,3,3])
output(tensor)



#indexing
print("-------------------------------------------")
print("\n\n\n")
A = np.zeros([5,5], dtype=np.int)
A[2,2] = 2
print(A)
print("\n")

A[:] = 3#tüm row ve col değerlere atar === A[:,:]
print(A)
print("\n")

A[:] = 3#tüm row ve col değerlere atar
print(A)
print("\n")

A[:, 0] = 4#tüm 0' ıncı kolonlara bu değeri atar.
print(A)
print("\n")



#for higher dimension simply add an index
print("-------------------------------------------")
print("\n\n\n")
A = np.ones([5,5,5], dtype=np.int)
A[:,0,0] = 6
print(A)



#matrix operations
print("-------------------------------------------")
print("\n\n\n")
A = np.matrix([[1,2],[3,4]])
B = np.ones([2,2], dtype=np.int)

print(A)
print()
print(B)

C = A+B
print()
print(C)

C = A-B
print()
print(C)


C = A*B
print()
print(C)


#matrix transpose
print("-------------------------------------------")
print("\n\n\n")
A = np.array(range(9))
A = A.reshape(3,3)
print(A)

B = A.T#transpose' u.
print()
print(B)

C = B.T#it's equals to orginal A.
print()
print(C)




print("\n\n\n")
A = np.array(range(10))
A = A.reshape(2,5)
print(A)

B = A.T
print()
print(B)




#tensors!
print("-------------------------------------------")
print("\n\n\n")
A = np.ones([3,3,3,3,3,3,3,3,3,3])
print(A.shape)
print(len(A.shape))
print(A.size)





















