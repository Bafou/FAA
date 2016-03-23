import numpy as np
from sklearn import tree
from sklearn import ensemble

def transforme_0_in_minus(vecteur):
	for donnee in vecteur:
		if donnee == 0 :
			donnee = -1
	return vecteur

def adaBoost(X,Y,nbTour):
	for i in range (nbTour):
		classifieur = tree.DecisionTreeClassifier()
		classifieur = classifieur.fit(X,Y)
	pass


#Main

diabet_data = np.loadtxt("pima-indians-diabetes.data", delimiter=",")
spam_data = np.loadtxt("spambase.data", delimiter=",")

#print(diabet_data.shape)
# separate the data from the target attributes
X = diabet_data[:,0:-1]
# Y = classe 
Y = diabet_data[:,-1]

taille= len(Y)
X_learn = X[0:(taille//10)*9,:]
X_test = X[(taille//10)*9:,:]
Y_learn = Y[:(taille//10)*9] 
Y_test =  Y[(taille//10)*9:]

print Y_test

print transforme_0_in_minus(Y_test)

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X_learn, Y_learn)

#print( clf.predict(X_test))

#tree.export_graphviz(clf, "test.dot")