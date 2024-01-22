import matplotlib.pyplot as plt
import mnist
import numpy as np
from scipy import sparse
import scipy.linalg as linalg
from utils import perturb_labels
from wse import wse
from get_graph import get_graph
import os


train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

x = np.reshape(train_images, (train_images.shape[0], -1))
y = train_labels
labels = np.unique(y)

inds = []
for l in labels:
    inds.append(np.argwhere(l == train_labels)[:, 0])
inds = np.concatenate(inds)
np.random.shuffle(inds)

x = x[inds, :]
y = y[inds]

lam = 0.3413
eta = 0.1
gamma = 0.9
num_clusters = len(labels)
epochs = 50

A = get_graph(x, k=20)

degs = A.sum(axis=1)
D = sparse.spdiags(degs.A.ravel(), 0, A.shape[0], A.shape[1])

L = D - A
L = sparse.csr_matrix(L)

eps = np.finfo(float).eps
type_lap = 'norm'
if type_lap == 'norm':
    # avoid dividing by zero
    degs[degs == 0] = eps
    # calculate inverse of D
    # Moore–Penrose inverse (D is diagonal)
    D = sparse.spdiags(1./degs.A.ravel(), 0, D.shape[0], D.shape[1])
    # calculate Left (random-walk) normalized Laplacian
    L = D.dot(L)

groups = perturb_labels(y, psnr=0.3)

n = L.shape[0]
W0 = linalg.orth(np.random.random(size=(n, num_clusters)))
Wt = W0
V = Wt

for epoch in epochs:
    L_tilde = sample(L)