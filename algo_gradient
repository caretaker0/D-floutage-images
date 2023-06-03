import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from imagebis import flou

photo = np.load('photo.npy') #on load une photo floutée représentée par un tableau npy en noir et blanc (c'est un vecteur de grande dimension)

plt.imshow(y) #on affiche une premiere fois la photo

#la fonction flou est linéaire, en effet :

X = np.random.randn(128,128)
Y = np.random.randn(128,128)

la.norm(flou(X)+flou(Y)-flou(X+Y)) #renvoie 1.9934665398239694e-14.

#Flou est de plus autoadjointe, en effet : 

X = np.random.randn(128,128)
Y = np.random.randn(128,128)

np.sum(X*flou(Y)) - np.sum(flou(X)*Y) #renvoie 0.0

#il suffit alors de minimiser l'expression ||flou(y) - photo||², ou y est un vecteur de meme dimension que photo.

#L'algo du gradient s'écrit :

def defloutage(y, x0, rho, itermax, tol):
    x = x0
    gradient = flou(flou(x)-y)
    for n in range(itermax):
        x = x - rho*gradient
        gradient = flou(flou(x)-y)
        if la.norm(gradient) < tol:
            break
    print(f"Nombre d'itérations effectuées: {n+1}")
    return x  
    
#on teste la fonction
x0 = np.zeros((128,128))
rho = 0.01
itermax=20000
tol = 10**(-4)
x = defloutage(y, x0, rho, itermax, tol)
    
#on affiche :

_ = plt.figure(dpi=100)
_ = plt.subplot(1,2,1)
plt.imshow(y) # l'image floue
_ = plt.subplot(1,2,2)
plt.imshow(x) # l'image renvoyée par l'algo 

#l'image est encore floue mais on peut trouver le tol optimal :

u = np.random.randn(128,128)
for _ in range(100):
    v = flou(u)
    u = v/la.norm(v)
    lmbda = np.sum( flou(u)*u )
lmbda
#renvoie 0.5486569635504208

x0 = np.zeros((128,128))
rho = 1/lmbda**2
itermax=10000
tol = 10**(-6)
x = defloutage(y, x0, rho, itermax, tol)

#on affiche :

_ = plt.figure(dpi=100)
_ = plt.subplot(1,2,1)
plt.imshow(y) # l'image floue
_ = plt.subplot(1,2,2)
plt.imshow(x) # l'image renvoyée par l'algo 

