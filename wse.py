import numpy as np
from sklearn.cluster import KMeans
from objective_wse import objective_wse
from objective_QL import objective_QL
import scipy.linalg as linalg


def wse(L, Wt, V, groups, maxiter=1000, tol=1e-5, eta=0.01, gamma=0.01, lam=0.3, num_clusters=2, W_update_type='closed_form'):
    L = L.toarray()
    a0, t = 1., 1
    obj = np.zeros(maxiter)
    n = Wt.shape[0]
    Ct = {}
    is_convergence = False
    G = np.unique(groups)
    a_prev = a0
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    # ------ adam optimizer parameters ------
    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    amsgrad = True
    m = np.zeros((Wt.shape[0], Wt.shape[1]), dtype=np.float32)
    v = np.zeros((Wt.shape[0], Wt.shape[1]), dtype=np.float32)
    vt_max = np.zeros((Wt.shape[0], Wt.shape[1]), dtype=np.float32)
    # ------ end of adam optimizer parameters  ------

    while t < maxiter and not is_convergence:
        obj[t] = objective_wse(Wt, groups, L, lam)

        if t > 2:
            if np.abs(obj[t] - obj[t-1]) < tol:
                print("converged")
                kmeans.fit(Wt)
                Ct[t] = kmeans.labels_
                is_convergence = True
                break
            elif obj[t] > obj[t-1]:
                print(t, obj[t], obj[t-1])
                print("obj started increasing")
                is_convergence = True
                break
        cost_q = objective_QL(Wt, V, L, groups, lam, eta)

        if obj[t] > cost_q:
            eta = gamma * eta

        W_prev = Wt

        V = Wt - eta*(2.*L.dot(Wt))
        # update Wt
        if W_update_type == 'standard':
            for i in range(len(G)):
                g = G[i]
                g_inds = np.argwhere(groups == g)
                g_inds = g_inds[:, 0]
                n_g = len(g_inds)
                C_g = np.eye(n_g) - (1. / n_g) * np.ones((n_g, n_g))
                V_g = V[g_inds, :]
                Wt[g_inds, :] = linalg.inv(np.eye(n_g) + (2. * lam * eta * n / (len(G) * n_g)) * C_g, overwrite_a=True).dot(V_g)
        elif W_update_type == 'fast':
            for i in range(len(G)):
                g = G[i]
                g_inds = np.argwhere(groups == g)
                g_inds = g_inds[:, 0]
                n_g = len(g_inds)
                beta = 2. * lam * eta * n / (n_g * len(G))
                beta_a = 1. + beta
                beta_b = -beta / n_g
                sum_V = np.sum(V[g_inds, :], axis=0)
                rep_sum_V = np.repeat(sum_V[np.newaxis, :], repeats=len(g_inds), axis=0)
                diff = (1. / beta_a) * V[g_inds, :] - beta_b / (beta_a * (beta_a + beta_b * n_g)) * rep_sum_V
                Wt[g_inds, :] = diff

        # sgd
        # Wt = W_prev - gamma * Wt

        # adaptive sgd
        at = 2. / (t + 3.)
        Wt = Wt + (1 - a_prev) / a_prev * at * (Wt - W_prev)
        a_prev = at
        Wt = linalg.orth(Wt)

        # Adam
        # gradient
        # delta = Wt - W_prev
        # m = beta1 * m + (1 - beta1) * delta
        # v = beta2 * v + (1.0 - beta2) * delta ** 2
        # mhat = m / (1.0 - beta1 ** (t + 1))
        # vhat = v / (1.0 - beta2 ** (t + 1))
        # if amsgrad:
        #     vt_max = np.maximum(vt_max, vhat)
        #     Wt = Wt - gamma * mhat / (np.sqrt(vt_max) + eps)
        # else:
        #     Wt = Wt - gamma * mhat / (np.sqrt(vhat) + eps)
        # Wt = linalg.orth(Wt)

        if (t + 1) % 1 == 0:
            print(t, obj[t], cost_q)

        # get clusters
        if (t + 1) % 50 == 0:
            kmeans.fit(Wt)
            Ct[t] = kmeans.labels_

        t += 1

    return obj, Ct, Wt

