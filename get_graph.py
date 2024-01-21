import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

# https://stackoverflow.com/questions/26576524/how-do-i-transform-a-scipy-sparse-matrix-to-a-numpy-matrix


def sim_gaussian(x, sigma):
    inds = x.nonzero()
    d = np.exp((-x.toarray()**2 / (2 * sigma ** 2)))
    knndist = coo_matrix((d[inds], inds), shape=(x.shape[0], x.shape[1]))
    return knndist


def get_graph(X, k, graph_type='mutual_knn'):
    dist = squareform(pdist(X))
    n = dist.shape[0]
    isnn = np.zeros((n, n), dtype=bool)

    # Create directed neighbor graph
    for iRow in range(n):
        idx = np.argsort(dist[iRow, :])
        isnn[iRow, idx[0:k]] = True

    # print("dist", dist[isnn])
    # print("isnn", isnn.nonzero())
    knndist = csr_matrix((dist[isnn], isnn.nonzero()), shape=(n, n))
    if graph_type == 'mutual_knn':
        knndist = knndist.minimum(knndist.transpose())
    elif graph_type == 'knn':
        knndist = knndist.maximum(knndist.transpose())

    sigma = np.median(knndist[isnn].A.ravel())  # Gaussian parameter
    A = sim_gaussian(knndist, sigma)

    return A


if __name__ == '__main__':
    # test data
    k = 2
    x = np.array(
        [[0.101840, -0.599612, -0.047494, -0.722723, 0.183573, 0.274240, 0.084235, -0.953521, -0.190030, -0.360554],
         [1.360070, -1.081047, 1.109594, -0.488779, -0.035228, 0.072544, 0.824734, 1.534444, -0.396467, -1.771091]]
    )
    x = x.transpose()
    A = get_graph(x,  k=k)
    print(A.A.ravel())

