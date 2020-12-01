####################################################################### 
#Modules importés
####################################################################### 
from math import *
#from tkinter import *
#from tkinter import messagebox
#import matplotlib.patches as mp
import pandas as pnd
import statistics as st
import numpy as np
import matplotlib.pyplot as pl
chemin = input("Veuillez spécifier l'adresse complète du fichier : ")
données = (open(chemin, mode='r')).readlines()

#C:\Users\ilies\OneDrive\Bureau\Downloads\EIVP_KM_actu.csv



#def csv_to_list(data,n): (A supprimer)
#    c = data[n]
#    l=[]
#    t=""
#    for x in c :
#            if x == ";" or x == "\n" :
#                l.append(t)
#                t=""
#            else :
#                t=t+x
#    return l
#######################################################################
#Extraction du fichier CSV et mise en forme des données
####################################################################### 
def extract(data,cat,sen):
    l = (data[0]).split(sep=";",maxsplit=-1)
    l[len(l)-1] = (l[len(l)-1]).replace("\n","")
    if cat not in l :
        print("Cette donnée n'est pas fournie.")
    else :
        k = l.index(cat)
        n = len(data)
        d=[]
        if cat == "sent_at" :
            for i in range(1,n):
                t = (((data[i]).split(sep=";",maxsplit=-1))[k]).split(" ")
                if int(((data[i]).split(sep=";",maxsplit=-1))[(l.index("id"))]) == sen :
                    d.append(pnd.Timestamp((t[0]+" "+(t[1].split("\n"))[0])))
        else :
            for i in range(1,n):
                if int(((data[i]).split(sep=";",maxsplit=-1))[(l.index("id"))]) == sen :
                    d.append(((data[i]).split(sep=";",maxsplit=-1))[k])
        return d
    
def interv(plage,tinf,tsup):
    j=0
    n=len(plage)
    while (j<n) and (plage[j] < pnd.Timestamp(tinf)):
        j=j+1
    i=j
    while (j<n) and (plage[j] < pnd.Timestamp(tsup)):
        j=j+1
    return [i,j]

def floatl(l):
    n = len(l)
    for i in range(n) :
        l[i]=float(l[i])
    return l
####################################################################### 
#Algorithmes de dessin
#######################################################################
def tracer(data,cat,sen,*args):
    plage=extract(data,"sent_at",sen)
    val=extract(data,cat,sen)
    if args == () :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = args[0]
        tsup = args[1]
    [i,j]=interv(plage,tinf,tsup)
    T=np.array(plage[i:j])
    X=np.array(floatl(val[i:j]))
    pl.plot(T,X,marker="+",linestyle='None',color=(0,0,0))
    pl.xticks(ticks=None)
    pl.legend(handles=[mp.Patch(label="Coefficient de corrélation : "+str(corrélation(données,"co2","lum",1,*args)))])
    pl.title("Graphique illustrant l'évolution de "+cat+" en fonction du temps entre le "+str(tinf)+" et le "+str(tsup))
    #pl.savefig('testgraph.pdf',format='pdf')
    pl.show()

def dessiner():
    data=input("Veuillez spécifier la base de données à analyser : ")
    cat=input("Quelle grandeur doit-être représentée ? ")
    sen=input("Quel capteur doit-être analysé ? ")
    tinf=input("A partir de quelle date souhaitez-vous initier la représentation ? ")
    tsup=input("A partir de quelle date souhaitez-vous arrêter la représentation ? ")
    tracer(data,cat,sen,tinf,tsup)
#######################################################################
#Fonctions statistiques
####################################################################### 
def moyenne_empirique(l):
    n=len(l)
    S=0
    for i in range(0,n):
        S=S+l[i]
    S=(1/n)*S
    return S

def variance(l):
    return st.variance(l)

def covariance(data,cat1,cat2,sen,*args):
    plage=extract(data,"sent_at",sen)
    if args == () :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = args[0]
        tsup = args[1]
    [i,j]=interv(plage,tinf,tsup)
    x=floatl((extract(data,cat1,sen))[i:j])
    y=floatl((extract(data,cat2,sen))[i:j])
    xm=moyenne_empirique(x)
    ym=moyenne_empirique(y)
    n=len(x)
    C=0
    for i in range(0,n):
        C = C + (x[i]-xm)*(y[i]-ym)
    C = C *(1/n)
    return C
    
def sigma(x):
    return sqrt(st.variance(x))

def corrélation(data,cat1,cat2,sen,*args):
    plage=extract(data,"sent_at",sen)
    if args == () :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = args[0]
        tsup = args[1]
    [i,j]=interv(plage,tinf,tsup)
    x=floatl((extract(data,cat1,sen))[i:j])
    y=np.array(floatl((extract(data,cat2,sen))[i:j]))
    sx=sigma(x)
    sy=sigma(y)
    C=covariance(data,cat1,cat2,sen,tinf,tsup)
    I=C/(sx*sy)
    return I
#######################################################################
#Mesures de similarité
####################################################################### 
def dtw(s, t):
    s=norm(s)
    t=norm(t)
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs((s[i-1] - t[j-1]))
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix
#s[0:i] et t[0:j]

def best_dtw(s,t):
    d = dtw(s,t)
    i = len(s)
    j = len(t)
    l = [(i,j)]
    while i>=1 and j>=1:
        mem=d[i-1][j]
        mi=i-1
        mj=j
        if d[i-1][j-1]<mem:
            mem=d[i-1][j-1]
            mj=mj-1
        elif d[i][j-1]<mem:
            mem=d[i][j-1]
            mi=mi+1
            mj=mj-1
        elif d[i-1][j-1] == d[i][j-1] and d[i-1][j] == d[i][j-1] :
            mj=mj-1
        i=mi
        j=mj
        l.insert(0,(i,j))
    return l

def dtwsim(data,cat,sen1,sen2,*args):
    plage1=extract(data,"sent_at",sen1)
    plage2=extract(data,"sent_at",sen2)
    if args == () :
        tinf1=min(plage1)
        tsup1=max(plage1)
        tinf2=min(plage2)
        tsup2=max(plage2)
    else :
        tinf1 = args[0]
        tsup1 = args[1]
        tinf2 = args[2]
        tsup2 = args[3]
    [i,j]=interv(plage1,tinf1,tsup1)
    [e,f]=interv(plage2,tinf2,tsup2)
    x=floatl((extract(data,cat,sen1))[i:j])
    y=floatl((extract(data,cat,sen2))[e:f])
    C=dtw(x,y)
    (n,m)=C.shape
    s=1-C[n-1][m-1]
    return s
    

def pcomp(data,cat,sen1,sen2,*args):
    plage1=extract(data,"sent_at",sen1)
    plage2=extract(data,"sent_at",sen2)
    if args == () :
        tinf1=min(plage1)
        tsup1=max(plage1)
        tinf2=min(plage2)
        tsup2=max(plage2)
    else :
        tinf1 = args[0]
        tsup1 = args[1]
        tinf2 = args[2]
        tsup2 = args[3]
    [i,j]=interv(plage1,tinf1,tsup1)
    [e,f]=interv(plage2,tinf2,tsup2)
    x=norm(floatl((extract(data,cat,sen1))[i:j]))
    y=norm(floatl((extract(data,cat,sen2))[e:f]))
    k=min(len(x),len(y))
    c=[]
    for i in range(k):
        c.append(abs(x[i]-y[i]))
    return c

def ratio(e,data,cat,sen1,sen2,*args):
    c=pcomp(data,cat,sen1,sen2,*args)
    d=0
    for i in c:
        if i>e:
            d=d+1
    return 1-(d/len(c))

def psim(data,cat,sen1,sen2,*args):
    c=pcomp(data,cat,sen1,sen2,*args)
    return(max(c),min(c),moyenne_empirique(c),sigma(c))
    
def paff(data,cat,sen1,sen2):
	 X=np.array(pcomp(data,cat,sen1,sen2))
	 T=np.array(list(range(0,len(X))))
	 pl.plot(T,X)
	 pl.axhline(psim(data,cat,sen1,sen2)[2],color='green')
	 pl.axhline(psim(data,cat,sen1,sen2)[0],color='red')
	 pl.axhline(psim(data,cat,sen1,sen2)[1],color='red')
	 pl.show()    
#######################################################################
#Fonctions auxiliaires
####################################################################### 
def norm(l):
    n,M,m=len(l),max(l),min(l)
    if M != m:
        for i in range(n):
            l[i]=(l[i]-m)/(M-m)
        return l
    else :
        for i in range(n):
            l[i]=l[i]/m
        return l

def pondere(l):
    n=len(l)
    val=[]
    freq=[]
    for i in l :
        if i not in val :
            val.append(i)
            freq.append(1/n)
        else:
            j=val.index(i)
            freq[j]=freq[j]+1/n
    return (val,freq)
    #P=[]
    #k=len(val)
    #for i in range(k):
    #    P.append(freq[i]*val[i])
    #return P
#######################################################################
#Algorithmes de correction
####################################################################### 
def lissage(k,l):
    # k impair
    n=len(l)
    R=[]
    for i in range((k+1)//2,n-(k+1)//2):
        M=l[i]/k
        for j in range(1,(k+1)//2):
            M = M + l[i+j]/k + l[i-j]/k
        R.append(M)
    return R
#######################################################################   
#def action_stat(l,f):
#    m=[]
#    n=len(l)
#    for i in range(n):
#        m.append(f(l[0:i+1]))
#    return m
#######################################################################
#Algorithmes d'automatisation de dessin et de sauvegarde
####################################################################### 
def load_all_courbes(data,tinf,tsup,lsen):
    X=[]
    T=[]
    l = (data[0]).split(sep=";",maxsplit=-1)
    l[len(l)-1] = (l[len(l)-1]).replace("\n","")
    k=l.index("sent_at")
    l.remove("sent_at")
    for i in lsen:
        T.append(np.array(interv(extract(data,"sent_at",i),tinf,tsup)))
        M=[]
        for j in l:
            M.append(np.array(floatl(extract(data,j,i))))
        X.append(M)
    return [T,X]

def save_all_courbes(data,lsen):
    l = (data[0]).split(sep=";",maxsplit=-1)
    l[len(l)-1] = (l[len(l)-1]).replace("\n","")
    k=l.index("sent_at")
    l.remove("sent_at")
    l.remove("")
    l.remove("id")
    for i in lsen: 
        T = np.array(extract(data,"sent_at",i))
        for j in l:
            X = np.array(floatl(extract(data,j,i)))
            pl.plot(T,X)
            pl.savefig('testgraph_'+j+'_'+str(i)+'_.pdf',format='pdf')
            pl.cla()
     

