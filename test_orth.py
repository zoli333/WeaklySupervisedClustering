import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from scipy.io import loadmat
import numpy as np
from scipy import sparse
import scipy.linalg as linalg
from utils import perturb_labels
from wse import wse
from get_graph import get_graph


Wt = np.random.random((5,5))
Wt = np.array([[0.03009866,0.91263986,0.2397577 ,0.82168971,0.37009999],
 [0.48065917,0.50186932,0.48086129,0.65396428,0.35997629],
 [0.20169533,0.59244539,0.0511279 ,0.20736873,0.71609986],
 [0.01968095,0.89119943,0.32681772,0.10619723,0.94754866],
 [0.63835154,0.39065753,0.71658693,0.37979068,0.00434365]])
print(Wt)
o = linalg.orth(Wt)
print(o)
o2 = np.linalg.qr(Wt)
print(o2)

groups = loadmat("../../Documents/WSC/results/graph/wlbl.mat", mat_dtype=True)
groups = groups['wlbl']
print(groups.shape)
inds = np.argwhere(groups == 1)
print(inds)
print(inds[:, 0])
print(inds[:, 0].shape)
print(inds.shape)