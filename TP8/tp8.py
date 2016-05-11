"""
PETIT Antoine & WISSOCQ Sarah

TP8 - Ada Boost
"""
import numpy as np
from sklearn import tree
from sklearn import ensemble
import matplotlib.pyplot as plt


def transforme_0_in_minus(vecteur):
    for donnee in vecteur:
        if donnee == 0 :
            donnee = -1
    return vecteur

#Tentative d'implementation de l'algorithme d'Adaboost
def adaBoost(X,Y,W,nbIteration):
    lenX = len(X)
    Wt = np.copy(W)
    HT=[]
    for i in range (nbIteration):
        classifieur = tree.DecisionTreeClassifier(max_depth=1)
        classifieur = classifieur.fit(X,Y)
        tab = classifieur.predict(X)
        nbError = 0
        for i in range(len(tab)):
            if tab[i] == Y[i] :
                nbError += Wt[i]*1
        epsilon = nbError / lenX
        alpha = 0.5 * np.log((1 - epsilon)/epsilon)
        interm = []
        for i in range(lenX):
            interm.append(Wt[i]*np.exp(-alpha*Y[i]*tab[i]))
        sumInterm = sum(interm)
        for i in range(lenX) :
            Wt[i] = interm[i]/sumInterm
        HT.append((classifieur, alpha))
    return HT


#Main

if __name__ == "__main__":
    diabet_data = np.loadtxt("pima-indians-diabetes.data", delimiter=",")
    spam_data = np.loadtxt("spambase.data", delimiter=",")

    #print(diabet_data.shape)
    # separate the data from the target attributes
    X = diabet_data[:,0:-1]
    # Y = classe 
    Y = diabet_data[:,-1]

    Y = transforme_0_in_minus(Y)

    taille= len(Y)
    X_learn = X[0:(taille//10)*9,:]
    X_test = X[(taille//10)*9:,:]
    Y_learn = Y[:(taille//10)*9] 
    Y_test =  Y[(taille//10)*9:]

    affich_y = []
    for i in range(1,200):
        aclf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                        learning_rate = 1,
                         algorithm="SAMME",
                         n_estimators=i)
        aclf.fit(X_learn,Y_learn)
        nbError = 0
        prediction = aclf.predict(X_test)
        for i in range (len(X_test)):
            if prediction[i] != Y_test[i] :
                nbError += 1
        affich_y.append(nbError)

    plt.plot(list(xrange(1,200)),affich_y)
    plt.show()

    # Cas donnant 0 faux positif
    weight = {1: 0.25, 0: 2.}
    aclf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1, class_weight = weight),
                        learning_rate = 1,
                         algorithm="SAMME",
                         n_estimators=200)
    aclf.fit(X_learn,Y_learn)
    nbError = 0
    fauxPositif = 0
    prediction = aclf.predict(X_test)
    for i in range (len(X_test)):
        print "Attendu : " , Y_test[i] , "; Obtenu : " , prediction[i]
        if prediction[i] != Y_test[i] :
            nbError += 1
        if prediction[i] == 1 and Y_test[i] == 0:
            fauxPositif += 1
    print  "Nombre de faux positif : ", fauxPositif
    taux = float(nbError)/float(len(Y_test))
    print "Taux d'erreur :" , taux
