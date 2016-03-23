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
	C = 4000.
	alpha = A / (B + C * float(t))
	return alpha

def descenteGradient(teta,x,y,N,epsilon=0.001):
	t=1

	list_temps = []
	list_mesureNormal2 = []

	list_temps.append(t)
	list_mesureNormal2.append(mesureNormal2(x,teta,y,N))
	tetaActuel = teta
	alpha = calcAlpha(t)
	p = y - np.dot(x.T,tetaActuel) # parenthese
	inte=np.dot(x,p)
	tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	while (math.fabs(mesureNormal2(x,tetaActuel,y,N) - mesureNormal2(x,tetaPlusPlus,y,N)) > epsilon):
		tetaActuel=tetaPlusPlus
		t=t+1
		list_temps.append(t)
		list_mesureNormal2.append(mesureNormal2(x,tetaActuel,y,N))
		alpha = calcAlpha(t)
		p = y - np.dot(x.T,tetaActuel) # parenthese
		inte=np.dot(x,p)
		tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	list_temps.append(t+1)
	list_mesureNormal2.append(mesureNormal2(x,tetaPlusPlus,y,N))
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

x0=np.array(lire('x0.txt'))
y0=np.array(lire('y0.txt'))
N=100
x=np.linspace(4,15,N)
mat0_1= creaMatricePuissanceM(x0,N,1)
mat0_10 = creaMatricePuissanceM(x0,N,10)
mat0_20 = creaMatricePuissanceM(x0,N,20)


m0_1 = moindreCarre(mat0_1,y0)
m0_10 = moindreCarre(mat0_10,y0)
m0_20 = moindreCarre(mat0_20,y0)

aff0_1 = matToFonc(m0_1,x0)
aff0_10 = matToFonc(m0_10,x0)
aff0_20 = matToFonc(m0_20,x0)


x1=np.array(lire('x1.txt'))
y1=np.array(lire('y1.txt'))
N1=300
mat1_1 = creaMatricePuissanceM(x1,N1,1)
mat1_10 = creaMatricePuissanceM(x1,N1,10)
mat1_20 = creaMatricePuissanceM(x1,N1,20)

m1_1 = moindreCarre(mat1_1,y1)
m1_10 = moindreCarre(mat1_10,y1)
m1_20 = moindreCarre(mat1_20,y1)


aff1_1 = matToFonc(m1_1,x1)
aff1_10 = matToFonc(m1_10,x1)
aff1_20 = matToFonc(m1_20,x1)


x2=np.array(lire('x2.txt'))
y2=np.array(lire('y2.txt'))
x2.sort()
y2.sort()
N2=300
mat2_1 = creaMatricePuissanceM(x2,N2,1)
mat2_10 = creaMatricePuissanceM(x2,N2,10)
mat2_20 = creaMatricePuissanceM(x2,N2,20)


m2_1 = moindreCarre(mat2_1,y2)
m2_10 = moindreCarre(mat2_10,y2)
m2_20 = moindreCarre(mat2_20,y2)

aff2_1 = matToFonc(m2_1,x2)
aff2_10 = matToFonc(m2_10,x2)
aff2_20 = matToFonc(m2_20,x2)

# Affichage des donnees

plt.close('all')
#Affichage jeu de donne 1
plt.plot(x0,y0,'.',color="red")
plt.plot(x0,aff0_1)
plt.title('Tp4 : Jeu de donnee 1; Polynome de degre 1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(x0,y0,'.',color="red")
plt.plot(x0,aff0_10)
plt.title('Tp4 : Jeu de donnee 1; Polynome de degre 10')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(x0,y0,'.',color="red")
plt.plot(x0,aff0_20)
plt.title('Tp4 : Jeu de donnee 1; Polynome de degre 20')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#Affichage jeu de donne 2

plt.plot(x1,y1,'.',color="red")
plt.plot(x1,aff1_1)
plt.title('Tp4 : Jeu de donnee 2; Polynome de degre 1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(x1,y1,'.',color="red")
plt.plot(x1,aff1_10)
plt.title('Tp4 : Jeu de donnee 2; Polynome de degre 10')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(x1,y1,'.',color="red")
plt.plot(x1,aff1_20)
plt.title('Tp4 : Jeu de donnee 2; Polynome de degre 20')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Affichage jeu de donne 3

plt.plot(x2,y2,'.',color="red")
plt.plot(x2,aff2_1)
plt.title('Tp4 : Jeu de donnee 3; Polynome de degre 1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(x2,y2,'.',color="red")
plt.plot(x2,aff2_10)
plt.title('Tp4 : Jeu de donnee 3; Polynome de degre 10')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.plot(x2,y2,'.',color="red")
plt.plot(x2,aff2_20)
plt.title('Tp4 : Jeu de donnee 3; Polynome de degre 20')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

