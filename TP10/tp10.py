"""
TP10 - SVM
"""

from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

##### Premieres donnees #####
X, Y = datasets.make_blobs(centers=2,n_samples=500, cluster_std=0.3, random_state=0)
clf = svm.SVC(C=1.0, kernel='linear',shrinking=False, probability=False,max_iter=500)

##### Deuxiemes donnees #####
#X, Y = datasets.make_blobs(centers=2,n_samples=500, cluster_std=0.3, random_state=0)
#clf = svm.SVC(C=0.01, kernel='linear',shrinking=False, probability=False,max_iter=500)

##### Troisieme donnees #####
#X, Y = datasets.make_blobs(centers=2,n_samples=500, cluster_std=0.8, random_state=0)
#clf = svm.SVC(C=1.0, kernel='linear',shrinking=False, probability=False,max_iter=500)

##### Quatrieme donnees ####
X, Y = datasets.make_circles(centers=2,n_samples=500, shuffle=True, noise=0.05,factor=0.5, random_state=0)
"""
clf.fit(X,Y)

hY = clf.predict(X)
error = np.mean(abs(hY - Y))

w = clf.coef_[0]
pente = -w[0] / w[1]

x = np.arange(0,5)
y = pente*x -clf.intercept_[0] / w[1]


b = clf.support_vectors_[0]
yy_down = pente * x + (b[1] - pente * b[0])
b = clf.support_vectors_[-1]
yy_up = pente * x + (b[1] - pente * b[0])
"""

plt.scatter(X[:,0], X[:,1], c=Y)
"""
plt.plot(x,y)
plt.plot(x,yy_down,color="red")
plt.plot(x,yy_up,color="red")

for point in clf.support_vectors_:
	plt.plot(point[0], point[1], 'cs')
"""
plt.show()