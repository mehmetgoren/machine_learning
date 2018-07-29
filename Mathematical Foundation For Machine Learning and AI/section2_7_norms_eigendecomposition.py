import numpy as np
from numpy import linalg

A = np.arange(9) - 3

B = A.reshape([3,3])

#vektörüm uzunluğunu bulmak için Euclşidean (L2) norm - default

r1 = np.linalg.norm(A)
r2 = np.linalg.norm(B)

#the max norm (P = infinity)
r3 = np.linalg.norm(A, np.inf)
r4 = np.linalg.norm(B, np.inf)

# vector normalization. norm burada uzunluk demek.
norm = np.linalg.norm(A)
A_unit = A / norm

# the magnitude of a unit vector is equal to 1
is_it_one = np.linalg.norm(A_unit)


# find the eigenvalues and eigenvectors fro a simple square matrix
A = np.diag(np.arange(1,4))#diagonal matrix oluşturur
eigenvalues, eigenvectors = np.linalg.eig(A)

#verify eigen decomposition
matrrix = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))


