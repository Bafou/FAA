import matplotlib.pyplot as plt
import numpy as np
import math

# np.loadtxt
def lire(chemin):
	fichier=open(chemin)
	res = []
	for ligne in fichier:
		res.append(float(ligne))
	fichier.close()
	return res
	
def lire_fichier_poids(chemin) :
    fichier = open(chemin,'r')
    (dim1,dim2) = ([],[])
    for ligne in fichier :
        (taille,poids) = ligne.split()
        dim1.append(float(taille))
        dim2.append(float(poids))
    return (dim1,dim2) 

def mesureAbs(x,teta,y,N):
	vecteur = y - np.dot(x.T,teta)
	return np.sum(np.absolute(vecteur))/N

def mesureNormal1(x,teta,y,N):
	vecteur = y - np.dot(x.T,teta)
	return math.sqrt(np.dot(vecteur.T, vecteur))/N

def mesureNormal2(x,teta,y,N):
	vecteur = y - np.dot(x.T,teta)
	return np.dot(vecteur.T, vecteur)/N


def mesureLinf(x,teta,y):
	vecteur = y - np.dot(x.T,teta)
	return np.amax(np.absolute(vecteur))

def moindreCarre(x,y):
	p1 = np.linalg.inv(np.dot(x,x.T))
	p2 = np.dot(x,y)
	return  np.dot(p1,p2)

def calcAlpha(t,valDef=1.):
	A = valDef	
	B = A
	C = 1.0
	alpha = A / (B + C * float(t))
	return alpha
	
def calcul_theta_suivant_sigmoid(iteration,theta):
    # Calcul du terme X(Y - X^T*θ_prec)
    tmp = np.dot(x, y - sigmoid(np.dot(x.T, theta)))

    # Calcul de θ = θ_prec + α_t * 1/N * X(Y - X^T*θ_prec)
    return np.add(theta, np.dot(calcAlpha(iteration) * (1.0 / n), tmp))
    
def sigmoid(X):
    A = 0.95
    b = -np.mean(X[0])

    return 1.0 / (1.0 + np.exp((np.dot(A, X) + b)))

def descenteGradientSigmoide(teta,x,y,N,epsilon=0.001):
    t=1

	list_temps = []
	list_mesureNormal2 = []

	list_temps.append(t)
	list_mesureNormal2.append(mesureNormal2sigmo(x,teta,y,N))
	tetaActuel = teta
	alpha = calcAlpha(t)
	p = y - sigmoide(np.dot(x.T,tetaActuel)) # parenthese
	inte=np.dot(x,p)
	tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	while (math.fabs(mesureNormal2sigmo(x,tetaActuel,y,N) - mesureNormal2sigmo(x,tetaPlusPlus,y,N)) > epsilon):
		tetaActuel=tetaPlusPlus
		t=t+1
		list_temps.append(t)
		list_mesureNormal2.append(mesureNormal2sigmo(x,tetaActuel,y,N))
		alpha = calcAlpha(t)
		p = y - sigmoide(np.dot(x.T,tetaActuel)) # parenthese
		inte=np.dot(x,p)
		tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	list_temps.append(t+1)
	list_mesureNormal2.append(mesureNormal2sigmo(x,tetaPlusPlus,y,N))
	plt.close('all')
	plt.plot(list_temps,list_mesureNormal2)
	plt.show()


	return tetaPlusPlus

def creaMatricePuissanceM(x,N,M):
	res = np.zeros((M +1 ,N))
	for i in range(0,M+1):
		res[i,:] = np.power(x,i)
	return res


def matToFonc(m,x):
	res=0
	for i in range(0,len(m)):
		res = res + m[i]* x**i
	return res


# Recuperation donnees



f = np.loadtxt("taillepoids_f.txt") # Tailles et poids des femmes
h = np.loadtxt("taillepoids_h.txt") # Tailles et poids des hommes

nb_f = len(f)
nb_h = len(h)

f_taille = f[:, 0]  # Tailles des femmes
h_taille = h[:, 0]  # Tailles des hommes

f_poids = f[:, 1]   # Poids des femmes
h_poids = h[:, 1]   # Poids des hommes

x = np.append(f_taille, h_taille)               # Tailles
y = np.append(np.ones(nb_f), np.zeros(nb_h))    # Classes (1 pour femme, 0 pour homme)

x = np.vstack((x, np.ones(len(x))))             

n = len(x)

teta = np.array([2,3], float)

grad =  descenteGradientSigmoide(teta, x, y, len(x))


print grad


# Affichage des donnees
"""
plt.close('all')
plt.ylim(-1, 2)
plt.plot(classe0,zeros,marker='o',color="blue")
plt.plot(classe1,ones,marker='v',color="red")
plt.title('Tp5')
plt.show()
"""


