import numpy as np
import scipy.linalg as linalg


def objective_QL(W, Wt, L, groups, lam, eta):
    G = np.unique(groups)
    psi_g = np.zeros(len(G))
    n = Wt.shape[0]
    for i in range(len(G)):
        g = G[i]
        g_inds = np.argwhere(groups == g)
        g_inds = g_inds[:, 0]
        n_g = len(g_inds)
        C_g = np.eye(n_g) - (1. / n_g) * np.ones((n_g, n_g))
        psi_g[i] = (n / (len(G) * n_g)) * np.trace(W[g_inds, :].T.dot(C_g).dot(W[g_inds, :]))

    psi = psi_g.sum()

    diff = (W - Wt)
    f_W = np.trace(Wt.T.dot(L).dot(Wt))
    comp1 = np.trace(diff.T.dot(2.*L.dot(Wt)))
    comp2 = linalg.norm(diff) ** 2 / (2. * eta)
    comp3 = lam * psi
    return f_W + comp1 + comp2 + comp3


