## Learning Facial Action Units from Web Images with Scalable Weakly Supervised Clustering
A Python implementation
Implements train_toy.py where the toy dataset and makemoons (sklearn) generated datasets can be runned.
Also now support mnist (mnist package installation required) training.

## Requirements
- scipy
- numpy
- python 3.*
- sklearn
- mnist (optional for train_mnist.py)

## Running the repo
Run either of these files
  train_toy.py (default)
  train_mnist.py (mnist package required)
  train_olivetti_faces.py (faces database - https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html)

## Result (on toy dataset)

![result.png](https://github.com/zoli333/WeaklySupervisedClustering/blob/main/result.png)

## References
Article
### https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Learning_Facial_Action_CVPR_2018_paper.pdf
Original implementation in Matlab:
### https://github.com/zkl20061823/WSC

# Bonus:
Adam Optimizer added
