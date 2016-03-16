import numpy as np
from sklearn import tree


#Main

diabet_data = np.loadtxt("pima-indians-diabetes.data", delimiter=",")
spam_data = np.loadtxt("spambase.data", delimiter=",")

print(diabet_data.shape)
# separate the data from the target attributes
X = diabet_data[:,0:-1]
# Y = classe 
Y = diabet_data[:,-1]

taille= len(Y)
X_learn = X[0:(taille//10)*9,:]
X_test = X[(taille//10)*9:,:]
Y_learn = Y[:(taille//10)*9] 
Y_test =  Y[(taille//10)*9:]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_learn, Y_learn)

tree.export_graphviz(clf, "test.dot")