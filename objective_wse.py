import numpy as np


def objective_wse(Wt, groups, L, lam):
    G = np.unique(groups)
    f_W = (Wt.T.dot(L).dot(Wt)).trace()
    psi_g = np.zeros(len(G))
    n = Wt.shape[0]
    for i in range(len(G)):
        g = G[i]
        g_inds = np.argwhere(groups == g)
        g_inds = g_inds[:, 0]
        n_g = len(g_inds)
        C_g = np.eye(n_g) - (1. / n_g) * np.ones((n_g, n_g))
        psi_g[i] = n / (len(G) * n_g) * np.trace(Wt[g_inds, :].T.dot(C_g).dot(Wt[g_inds, :]))

    psi = psi_g.sum()
    return f_W + lam * psi