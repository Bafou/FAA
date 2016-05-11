"""
TP10 - SVM
Donnees non lineaire
"""

from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


##### Quatrieme donnees ####
X, Y = datasets.make_circles(n_samples=500, shuffle=True, noise=0.05,factor=0.5, random_state=0)

clf = svm.NuSVC()
clf.fit(X,Y)

Z= clf.decision_function(np.c_[X, Y])
#Z= Z.reshape(X.shape)

plt.imshow(Z, interpolation = 'nearest', extent=( X.min(), X.max()) , aspect = 'auto', origin = 'lower', cmap = lt.cm.PuOr_r)
plt.scatter(X[:,0], X[:,1], c=Y)

plt.show()