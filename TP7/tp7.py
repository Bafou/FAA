"""
PETIT Antoine & WISSOCQ Sarah

TP7 Python Fondements de l'Apprentissage Automatique
Arbre de decision - Bagging

"""

import sys
import numpy as np
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier




#Main
if __name__ == "__main__":

	try :
		inputfile = sys.argv[1]
		profondeur_arbre_max = int(sys.argv[2])
	except IndexError :
		print "Pour utiliser le programme il faut entrer la commande suivante : python2 tp7.py <nom_du_fichier> <profondeur_arbre_max>"
		sys.exit()
	except ValueError : 
		print "Attention la taille doit etre un entier"
		sys.exit()

	data = np.loadtxt(inputfile, delimiter=",")
	print(data.shape)
	# separate the data from the target attributes
	X = data[:,0:-1]
	# Y = classe 
	Y = data[:,-1]

	taille= len(Y)

	#### Extraction des donnees 90% pour l'apprentissage et 10% pour les tests
	X_learn = X[0:(taille//10)*9,:]
	X_test = X[(taille//10)*9:,:]
	Y_learn = Y[:(taille//10)*9] 
	Y_test =  Y[(taille//10)*9:]

	print "Decision Tree Classifier :"
	for profond in range (1,profondeur_arbre_max):
		clf = tree.DecisionTreeClassifier(max_depth = profond)
		clf = clf.fit(X_learn, Y_learn)
		nbError = 0
		prediction = clf.predict(X_test)
		for i in range (len(X_test)):
			if prediction[i] != Y_test[i] : 
				nbError += 1
		taux = float(nbError)/float(len(Y_test))
		#print prediction
		print "pour une profondeur " , profond, "on a un taux d'erreur egal a " , taux


	print "Random Forest Classifier :"
	for profond in range (1,profondeur_arbre_max):
		rfcl = RandomForestClassifier (n_estimators = 10, max_depth=profond)
		rfcl = rfcl.fit(X_learn,Y_learn)
		nbError = 0
		prediction = rfcl.predict(X_test)
		for i in range (len(X_test)):
			if prediction[i] != Y_test[i] :
				nbError += 1
		taux = float(nbError)/float(len(Y_test))
		#print prediction
		print "pour une profondeur ", profond, "on a taux d'erreur egal a " , taux

	print "Extremely Randomized Forest Classifier :"
	for profond in range (1,profondeur_arbre_max):
		erfcl = ExtraTreesClassifier (n_estimators = 10, max_depth=profond)
		erfcl = erfcl.fit(X_learn,Y_learn)
		nbError = 0
		prediction = erfcl.predict(X_test)
		for i in range (len(X_test)):
			if prediction[i] != Y_test[i] :
				nbError += 1
		taux = float(nbError)/float(len(Y_test))
		#print prediction
		print "pour une profondeur ", profond, "on a taux d'erreur egal a " , taux


	print "__________________Influence du nombre des arbres sur moyenne et variance__________________"

	print "Random Forest Classifier :"
	listmoyennes = []
	listvariances = []
	for arbre in range(5,25) :
		listTaux = []
		for stat in range(30) :
			rfcl = RandomForestClassifier(n_estimators=arbre,max_depth=5)
			rfcl = rfcl.fit(X_learn,Y_learn)
			nbError = 0
			prediction = rfcl.predict(X_test)
			for i in range(len(X_test)) :
				if prediction[i] != Y_test[i] :
					nbError = nbError + 1
				tauxErr = float(nbError)/float(len(Y_test))
				listTaux.append(tauxErr)
		listmoyennes.append((arbre,np.mean(listTaux)))
		listvariances.append((arbre,np.var(listTaux)))

	for (arbre,moyenne) in listmoyennes :
		print "La moyenne des erreurs pour un nombre d'arbre de  ",arbre," est de ",moyenne

	for (arbre,variance) in listvariances :
		print "La variance des erreur pour un nombre d'arbre de ",arbre," est de ",variance


	print "Extremely Randomized Forest Classifier :"

	listmoyennes = []
	listvariances = []
	for arbre in range(5,20) :
		listTaux = []
		for stat in range(30) :
			ExtraTreeCL = ExtraTreesClassifier(n_estimators=arbre,max_depth=5)
			ExtraTreeCL = ExtraTreeCL.fit(X_learn,Y_learn)
			nbError = 0
			prediction = ExtraTreeCL.predict(X_test)
			for i in range(len(X_test)) :
				if prediction[i] != Y_test[i] :
					nbError = nbError + 1
			tauxErr = float(nbError)/float(len(Y_test))
			listTaux.append(tauxErr)
		listmoyennes.append((arbre,np.mean(listTaux)))
		listvariances.append((arbre,np.var(listTaux)))

	for (arbre,moyenne) in listmoyennes :
		print "La moyenne des erreur pour un nombre d arbre de  ",arbre," = ",moyenne
	   
	for (arbre,variance) in listvariances :
		print "La variance des erreur pour un nombre d arbre de ",arbre," = ",variance

	#print( clf.predict(X_test))

	#tree.export_graphviz(clf, "test.dot")