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


# Recuperation donnees

t=np.array(lire('t.txt'))
p=np.array(lire('p.txt'))
a=2
b=3
N=100
x=np.linspace(4,15,N)
y=a*x+b

# Affichage des donnees

plt.close('all')
plt.plot(x,y)
plt.plot(t,p,'.')
plt.title('Tp1')
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.show()

# Calcul des performances

z = np.ones(len(x))

x1 = np.zeros((2, N))
x1[1,:] = t
x1[0,:] = z

t= np.array([b,a], float)

print "Jlabs =", mesureAbs(x1,t,p,N)
print "Jl1 =", mesureNormal1(x1,t,p,N)
print "Jl2 =", mesureNormal2(x1,t,p,N)
print "Jlinf =", mesureLinf(x1,t,p)

