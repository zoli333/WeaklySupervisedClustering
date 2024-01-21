from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import scipy.linalg as linalg
from utils import perturb_labels
from wse import wse
from get_graph import get_graph
import os


rng = RandomState(0)

faces, labels = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
n_samples, n_features = faces.shape

# centering

# Global centering (focus on one feature, centering all samples)
faces_centered = faces - faces.mean(axis=0)

# Local centering (focus on one sample, centering all features)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

y = np.bincount(labels)
ii = np.nonzero(y)[0]
print(np.vstack((ii,y[ii])).T)

x = faces_centered
y = labels


lam = 0.3413
eta = 0.1
gamma = 0.9
num_clusters = 40


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
cost, clusters, W = wse(L=L, Wt=Wt, V=V, maxiter=2000, groups=groups, eta=eta, gamma=gamma, lam=lam, num_clusters=num_clusters, W_update_type='fast')

last_t = list(clusters)[-1]
clusters = clusters[last_t]
c = np.unique(clusters)
for cluster in c:
    savename = "result_olivetti_faces/cluster_" + str(cluster)
    os.makedirs(savename, exist_ok=True)
    label = np.argwhere(clusters == cluster)
    label = label[:30]
    cluster_x = x[label, :]
    i = 0
    print("creating images for cluster: " + str(cluster))
    for img in cluster_x:
        img = img.reshape((64, 64))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(savename + '/img' + str(i) + '.png')
        i += 1
    print("-- finished creating images for cluster: " + str(cluster))
    plt.close()
plt.close()
