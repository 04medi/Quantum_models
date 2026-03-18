import numpy as np
import scipy 
import matplotlib.pyplot as plt

n=4
def bohr(n):
    Rh=-13.6 #eV
    E_bohr=np.array([])  #[n]
    for i in range(n):
        E_bohr=np.append(E_bohr,float(Rh*((1/(i+1)**2))))
        i+=1
        
    return E_bohr

def dirac(n):
    alfa=scipy.constants.fine_structure
    m=scipy.constants.electron_mass
    c=scipy.constants.c

    E_dirac=np.zeros((n,n))  #matrice dove [n,j]=[righe,colonne]

    for i in range(n):  # contatore n-1 (num quantici)
        N=i+1      #numeri quantici
        for J in range(i+1):  #contatore j 
            j=J+0.5       
            E_dirac[i,J]=6.242e+18*(m*c**2)*(-1+(1+(alfa/(N-j-0.5+np.sqrt(((j+0.5)**2)-alfa**2)))**2)**(-1/2))  #[eV]
            

    return E_dirac




colori = ['royalblue', 'red', 'green', 'yellow']

Liv_bohr=bohr(n)
Liv_dirac=dirac(n)

print(Liv_dirac)

# Figura per livelli Bohr
plt.figure(figsize=(5, 4))
for i in range(n):
    plt.axhline(Liv_bohr[i], color=colori[i], label=f'Livello {i+1}')
    for j in range(n):
        if(Liv_dirac[i,j]!=0):
            plt.axhline(Liv_dirac[i,j], color=colori[i], label=f'j= {j+0.5}')
plt.title('Livelli energetici di Bohr')
plt.ylabel('E(eV)')
plt.legend()
plt.grid(True)
plt.show()