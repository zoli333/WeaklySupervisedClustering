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
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

inds = []
for l in labels:
    inds.append(np.argwhere(l == train_labels)[:, 0])
inds = np.concatenate(inds)
np.random.shuffle(inds)

x = x[inds, :]
y = y[inds]

n = 1000
x = x[:n]
y = y[:n]

lam = 0.3413
eta = 0.1
gamma = 0.9
num_clusters = 10


A = get_graph(x, k=10)

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
cost, clusters, W = wse(L=L, Wt=Wt, V=V, maxiter=10000, groups=groups, eta=eta, gamma=gamma, lam=lam, num_clusters=num_clusters, W_update_type='fast')

last_t = list(clusters)[-1]
clusters = clusters[last_t]
c = np.unique(clusters)
for cluster in c:
    savename = "result/cluster_" + str(cluster)
    os.makedirs(savename, exist_ok=True)
    label = np.argwhere(clusters == cluster)
    label = label[:30]
    cluster_x = x[label, :]
    i = 0
    print("creating images for cluster: " + str(cluster))
    for img in cluster_x:
        img = img.reshape((28, 28))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(savename + '/img' + str(i) + '.png')
        i += 1
    print("-- finished creating images for cluster: " + str(cluster))
    plt.close()

plt.close()