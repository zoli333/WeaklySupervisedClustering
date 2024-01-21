import math
import numpy as np
import scipy.sparse
from scipy.spatial.distance import cdist
import scipy.sparse as sparse
from scipy.sparse import lil_matrix, linalg
import torch


x = np.array([[0,2,3],[5,2,0],[2,30,1]])
print(x)
y = sparse.csr_matrix(x)
print(y)
print("-------")
z = (-y.power(2) / (2*0.738**2))
print(z)
z = np.exp(z.A)
print(z)
print()

print("*********")
s = (-y**2 / (2*0.738**2))
print(np.exp(s.todense()))

print("/////////////////")
arr = torch.from_numpy(x)
b = arr.to_sparse_coo()
res = -b**2 / (2*0.738**2)
print(res)
c = np.exp(res.to_dense())
print(c)

# Matlab
# x = [0 2 3; 5 2 0; 2 30 1]
# x = sparse(x)
# z = -x.^2 ./ (2*0.738^2)
# z2 = exp(z)

print("-------------")
print(np.random.choice(10, size=30))

print("-------------")
row  = np.array([0, 0, 1, 3, 1, 0, 0])
col  = np.array([0, 2, 1, 3, 1, 0, 0])
data = np.array([1, 1, 1, 1, 1, 1, 1])
coo = sparse.coo_matrix((data, (row, col)), shape=(5, 5))
print(coo)