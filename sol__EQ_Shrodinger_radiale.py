import numpy as np
import scipy 
from scipy.constants import hbar  #h-tagliato h/2*pi [J*s]


#Il programma risolve l'equazione Schrodinger radiale per un qualsiasi Potenziale
#Si ipotizza che la matrice V sia già in coordinate(r)


#Definisci intervallo utile
m=            #massa particella interessata nell'orbitale 
R_max=    #Raggio massimo utile

N=            #Numero di discretizzazioni
L=            #numero momento angolare
r=np.linspace(1e-10,R_max,N)
l=[0,1,2,3]  #vettore momento angolare
toll=         #tolleranza del valore di numerov


#definisci V -> Potenziale da descrivere in base al problema

V= ####


#1° parte RICERCA AUTOVALORI PER DIAGONALIZZAZIONE
#Cioè si ricerca Autovalori Hu=Eu dove H= -a*D + Veff

#definisci Vefficace
Veff = lambda l, r: V + l*(l+1)*(hbar**2)/(2*m*r**2)

#costruzione matrice Laplaciano
h=r[1]-r[0]                          #definizione del passo

diag = np.ones(N)

D2 = (                              #matrice u''=(u(i+1)-2*ui + ui-1)/h**2
    np.diag(diag[:-1],-1)
    -2*np.diag(diag,0)
    +np.diag(diag[:-1],1)
)/h**2

a= (hbar**2)/(2*m) #costante 

#--------------------------------!!!!!!RICORDA DI Ciclare per i vari momenti angolari !!!!!
#Matrice Hamiltoniana
H=-a*D2 + np.diag(Veff(l[L],r))               

#ricerva autovalori e autovettori

E_d,U_d=np.linalg.eigh(H)    
# in Questo caso gli autovettori sono strutturati per U_d[:,k] per k-esimo Autovettore
# e i-esima cella nel r=[0,R] in k-esimo vettore -> U_d[i,k]                         


#-----------------------------------------
#2° Parte Ricerca Numerov + Shooting



#HP Si Ipotizza che gli autovalori (Energie) e i rispettivi nodi dei rispettivi Autovett siano sempre in crescente E1<E2<..<En
#a è il vettore di energie Dato da shooting
#E è il vettore energie dato dalla diagonalizzazione
#n è il numero di nodi della funzione ricercata

#Tollereanza assoluta 
def bisez(n,E_NS, E_d, toll):  

    #la bisezione trova il n-esimo autovett e autovalore tra il limite inferiore -a- (n-1) e il limite superiore -b- (n+1)
    #nel caso non sia presente un'autoval intermedio si estende l'intervallo per (n+k) con k il numero di iterazioni

    #Definizione limiti 
    if(E_d[0]<0):    #Energie negative(legame) Esempio: E=[-10,-5,-3,-1]


        if(n==0):      #Caso Fondamentale
            a=E_d[n]*2
            b=E_d[n+1]

        elif(n==len(E_d)-1): #Caso Ultima energia
            a=E_NS[n-1] #ultimo valore
            b=0

    elif(E_d[0]>0): #non legame   Esempio: E=[1,2,4,5]
        if(n==0):
            a=0
            b=E_d[n+1]
        elif(n==len(E_d)-1):
            a=E_NS[n-1] #ultimo valore
            b=b=2*E_d[n]-E_d[n-1]
    else:   
        b=E_d[n+1]
        a=E_NS[n-1] 


    
    #Ricalcolo Intervallo -Algoritmo di bisezione-
    u_b=numerov(b)    

    if(E_d[0]<0 and n==len(E_d)):     #Caso in cui b=0 cioè non determinabile
        None
    else:

        #INIZIALIZZAZIONE
        nb=nodi(u_b)  #calcolo nodo del limite superiore
    
        k=1  #contatore per estensione
        Estensione=True

        bk=b  #Limite superiori
        ak=a       #limite Inferiore
        xk=(a+b)/2

        #Attento In questo caso b non è lo stesso che useremo dopo
        while(nb!=n+1):  #Fino a quando il nodo corretto del limite superiore non viene indentificato

           
            if(nb<n+1):         #Caso Intervallo troppo piccolo->allargamento estendo secondo
                #La condizione Estensione Non si deve ripetere quando si è superato il numero di nodi nb>n+1 ed entra in gioco Bisezione
                if(Estensione==True and k==len(E_d)-1):   
                    bk=E_d[n+k]        
                    k+=1


                if(Estensione==False):
                    ak=xk
                    bk=bk

            elif(nb>n+1):  #Caso Intervallo troppo grande ->Rimpicciolimento
                Estensione=False
                bk=xk
                ak=ak

            elif(nb==n+1):
                b=xk
                break

            xk=(ak+bk)/2     #per trovare limite superiore
            nb=nodi(numerov(bk))

    
    #Calcolo Valore finali
    #Troviamo penultimo elemento delle funzioni
    F_a=u_a[-1]
    F_xk=u_x[-1]

    
    i=1
    xk=(a+b)/2 
    #1 Guess delle Autofunzioni
    u_x=numerov(xk)  # u è l'autovettore iniziale e invochiamo numerov per ottenere il valore
    u_a=numerov(a)
    #HP na<nx<nb
    while (F_xk!=0 and abs(F_xk)>toll):
        i+=1

        if(F_xk*F_a<0):

            b=xk
        else:
            a=xk
            u_a=numerov(a)
            F_a=u_a[-1]

        xk=(a+b)/2

        u_x=numerov(xk)
        F_xk=u_x[-1]

        if(i>=1e+3):    #contatore di non convergenza
            return xk,u_x

    return xk,u_x #ritorna l'autovalore e autovettore


def numerov (E):   # E è l'autovalore 
    r0=r[0]**(l[L]+1)
    r1=r[1]**(l[L]+1)
    Y=np.array([r0,r1])    #array soluzione, 

    n=len(r)     

    a= (hbar**2)/(2*m) #costante
    for j in range(2,n):     #la ricerca dell'algoritom di numerov parte da j=2  cioè j=i-1

        g_1=(Veff(l[L],r[j-2])-E)/a
        g=(Veff(l[L],r[j-1])-E)/a
        g_plus=(Veff(l[L],r[j])-E)/a

        y_plus_1=(2*Y[j-1]*(1+5*(h**2)*g/12)-(1-g_1*(h**2)/12)*Y[j-2])/(1-(h**2)*g_plus/12)

        Y=np.append(Y,y_plus_1)


    #Normalizzazione del autostato    int(u(r)**2)=1 tra [0,R]
    #Integrazione per Trapezio
    I=0.5*h*(Y[0]**2+Y[-1]**2)    

    for i in range(1,N-1):  #il contatore salta i valori ai bordi (già sommati prima)
        I=I+h*Y[i]**2

    Y=Y/np.sqrt(I)
    return Y   


def nodi(u):#u->vettore gia integrato, r 
    N=0
    for i in range(1,len(u)-1): # i è i-esimo+1vettore
        if(u[i]*u[i+1]<0):
            N+=1

    return N

#Applicazione Numerov+shooting
n=len(E_d) # numero di Autovalori 

#Inizializza nuovi Autovalori e Autovettori

E_NS=np.array([])

U_NS=np.zeros((len(r),n))  #matrice  dove gli autovettori sono strutturati per U_NS[:,k] per k-esimo Autovettore
# e i-esima cella nel r=[0,R] in k-esimo vettore -> U_NS[i,k]   

for i in range(n):      #i indica i-esimo autoval-autovett e k-esimo nodo

    e , u =bisez(i,E_NS,E_d,toll)

    E_NS=np.append(E_NS,e)
    U_NS[:,i]=u
























