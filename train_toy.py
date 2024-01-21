import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from scipy.io import loadmat
import numpy as np
from scipy import sparse
import scipy.linalg as linalg
from utils import perturb_labels
from wse import wse
from get_graph import get_graph

lam = 0.3413
eta = 0.1
gamma = 0.9
# gamma = 0.001 -- for Adam optimizer (but works fine with 0.9 as well)
num_clusters = 2

# moons dataset
# x, y = datasets.make_moons(n_samples=500, noise=0.1)

# original dataset
data = loadmat("data/toy.mat", mat_dtype=True)
x = data['feat'].transpose()
y = data['label'].squeeze()

A = get_graph(x, k=100)

degs = A.sum(axis=1)
D = sparse.spdiags(degs.A.ravel(), 0, A.shape[0], A.shape[1])

L = D - A
L = sparse.csr_matrix(L)

eps = np.finfo(float).eps
type_lap = 'norm'
if type_lap == 'norm':
    # avoid dividing by zero
    degs[degs == 0] = eps
    # calculate ingerse of D
    D = sparse.spdiags(1./degs.A.ravel(), 0, D.shape[0], D.shape[1])
    # % calculate normalized Laplacian
    L = D.dot(L)

groups = perturb_labels(y, psnr=0.3)

n = L.shape[0]
W0 = linalg.orth(np.random.random(size=(n, num_clusters)))
Wt = W0
V = Wt

# run wse
cost, clusters, W = wse(L=L, Wt=Wt, V=V, groups=groups, eta=eta, gamma=gamma, lam=lam, num_clusters=num_clusters, W_update_type='fast')

last_t = list(clusters)[-1]
plt.scatter(x[:, 0], x[:, 1], c=clusters[last_t])
plt.axis('off')
plt.savefig('result_toy.png')
plt.show()