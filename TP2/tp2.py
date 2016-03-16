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


# Recuperation donnees

t=np.array(lire('t.txt'))
p=np.array(lire('p.txt'))
a=2
b=3
N=100
x=np.linspace(4,15,N)
y=a*x+b

# Calcul des performances

z = np.ones(len(x))

x1 = np.zeros((2, N))
x1[1,:] = t
x1[0,:] = z



m=moindreCarre(x1,p)

y2= m[1]*x + m[0]

# Affichage 

t2 = np.array([b,a], float)
t3 = np.array(m, float)

ab1 = mesureAbs(x1,t2,p,N)
norm1_1 = mesureNormal1(x1,t2,p,N)
norm2_1 = mesureNormal2(x1,t2,p,N)
inf1 = mesureLinf(x1,t2,p)
ab2 = mesureAbs(x1,t3,p,N)
norm1_2 = mesureNormal1(x1,t3,p,N)
norm2_2 = mesureNormal2(x1,t3,p,N)
inf2 = mesureLinf(x1,t3,p)

print "-------------------Erreur-------------------------------------------"
print ""
print "Jlabs =", ab1
print "Jl1 =", norm1_1
print "Jl2 =", norm2_1
print "Jlinf =", inf1
print ""
print "-------------------Erreur avec moindreCarre-------------------------"
print ""
print "Matrice moindreCarre : ", m
print ""
print "Jlabs =", ab2
print "Jl1 =", norm1_2
print "Jl2 =", norm2_2
print "Jlinf =", inf2
print ""
print "-------------------Difference des erreurs---------------------------"
print ""
print "Diff(Jlabs1) =", ab1 - ab2
print "Diff(Jl1) =", norm1_1 - norm1_2
print "Diff(Jl2) =", norm2_1 - norm2_2
print "Diff(Jlinf) =", inf1 - inf2

# Affichage des donnees

plt.close('all')
line1, = plt.plot(x,y, label ='a*x + b', color="blue")
line2, = plt.plot(x,y2, label = 'Resultat moindreCarre',color="green")
plt.plot(t,p,'.',color="red")
plt.legend(handles = [line1,line2])
plt.title('Tp1')
plt.xlabel('Temps (s)')
plt.ylabel('Position (m)')
plt.show()


