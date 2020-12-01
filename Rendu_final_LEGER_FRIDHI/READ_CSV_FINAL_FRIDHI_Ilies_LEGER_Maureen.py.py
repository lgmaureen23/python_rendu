#C:\Users\ilies\OneDrive\Bureau\Downloads\EIVP_KM_actu.csv
####################################################################### 
#Modules importés
#######################################################################
All=["minimum","maximum","moyenne","médiane","quartile1","quartile3","variance","écart-type"]
from math import *
import sys as sys
import matplotlib.patches as mp
import pandas as pnd
import statistics as st
import numpy as np
import matplotlib.pyplot as pl
chemin = input("Veuillez spécifier l'adresse complète du fichier : ")
données = (open(chemin, mode='r')).readlines()
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
            if sen==None :
                for i in range(1,n):
                    t = (((data[i]).split(sep=";",maxsplit=-1))[k]).split(" ")
                    d.append(pnd.Timestamp((t[0]+" "+(t[1].split("\n"))[0]),tz=7200))
            else :
                for i in range(1,n):
                    t = (((data[i]).split(sep=";",maxsplit=-1))[k]).split(" ")
                    if int(((data[i]).split(sep=";",maxsplit=-1))[(l.index("id"))]) == sen :
                        d.append(pnd.Timestamp((t[0]+" "+(t[1].split("\n"))[0]),tz=7200))
        else :
            if sen==None :
                 for i in range(1,n):
                        d.append(float(((data[i]).split(sep=";",maxsplit=-1))[k]))
            else :
                for i in range(1,n):
                    if int(((data[i]).split(sep=";",maxsplit=-1))[(l.index("id"))]) == sen :
                        d.append(float(((data[i]).split(sep=";",maxsplit=-1))[k]))
        return d
    
def interv(plage,tinf,tsup):
    j=0
    n=len(plage)
    while (j<n) and (plage[j] < tinf):
        j=j+1
    i=j
    while (j<n) and (plage[j] < tsup):
        j=j+1
    return [i,j]
####################################################################### 
#Algorithmes de dessin
#######################################################################
def tracer(couleur,data,cat,sen,*btemp):
    plage=extract(data,"sent_at",sen)
    val=extract(data,cat,sen)
    if btemp == ((),()) or btemp ==() :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = pnd.Timestamp(btemp[0],tz=7200)
        tsup = pnd.Timestamp(btemp[1],tz=7200)
    [i,j]=interv(plage,tinf,tsup)
    T=np.array(plage[i:j])
    X=np.array(val[i:j])
    pl.plot(T,X,marker="+",linestyle='None',color=couleur)
    pl.xticks(ticks=None)
    pl.gca().legend()
    #pl.legend(handles=[mp.Patch(label="Coefficient de corrélation : "+str(action_stat("corrélation",data,cat,sen,*btemp)))])

def tracer_stat(lparam,data,cat,sen,*btemp):
    plage=extract(data,"sent_at",sen)
    val=extract(data,cat,sen)
    if btemp == ((),()) or btemp==() :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = pnd.Timestamp(btemp[0],tz=7200)
        tsup = pnd.Timestamp(btemp[1],tz=7200)
    [i,j]=interv(plage,tinf,tsup)
    X=np.array(val[i:j])
    for p in lparam:
        m=action_stat(p,data,cat,sen,*btemp)
        if p=="maximum" or p=="minimum":
            pl.axhline(m,color='red',linestyle='dashed')
            print(str(p)+" du capteur n°"+str(sen)+" : "+str(m))
        if p=="moyenne":
            pl.axhline(m,color='green',linestyle='dashed')
            print(str(p)+" du capteur n°"+str(sen)+" : "+str(m))
        if p=="quartile1" or p=="médiane" or p=="quartile3":
            pl.axhline(m,color='purple',linestyle='dashed')
            print(str(p)+" du capteur n°"+str(sen)+" : "+str(m))
        if p=="variance" or p=="écart-type":
            print(str(p)+" du capteur n°"+str(sen)+" : "+str(m))
    pl.xticks(ticks=None)
    pl.gca().legend()
#######################################################################
#Fonctions statistiques
#######################################################################
def minimum(l):
    if l == []:
         print("La liste est vide")
    else:
        m = l[0]
        n = len(l)
        for i in range(1,n):
            if l[i]<m:
                m = l[i]
        return m

def maximum(l):
    if l == []:
         print("La liste est vide")
    else:
        M = l[0]
        n = len(l)
        for i in range(1,n):
            if M<l[i]:
                M = l[i]
        return M
  
def moyenne(l):
    n=len(l)
    S=0
    for i in range(0,n):
        S=S+l[i]
    S=(1/n)*S
    return S

def variance(l):
    m=moyenne(l)
    n=len(l)
    E=0
    for i in range(n):
        E = E + (l[i]-m)**2
    return E/n

def covariance(x,y):
    xm=moyenne(x)
    ym=moyenne(y)
    n=len(x)
    C=0
    for i in range(0,n):
        C = C + (x[i]-xm)*(y[i]-ym)
    C = C *(1/n)
    return C
    
def sigma(x):
    return sqrt(variance(x))

def corrélation(x,y):
    sx=sigma(x)
    sy=sigma(y)
    C=covariance(x,y)
    I=C/(sx*sy)
    return I

def evol_distrib_freq(l):
    X=[]
    F=[]
    N=len(l)
    for i in l:
        if i in X:
            j=X.index(i)
            F[j]=F[j]+1/N
        else:
            X.append(i)
            F.append(1/N)
    X,F = arrange(X,F)
    return X,F

def médiane(l):
    X,F=evol_distrib_freq(l)
    S=0
    i=0
    while S < 0.5 :
        S = S + F[i]
        i = i + 1
    return X[i]

def quartile1(l):
    X,F=evol_distrib_freq(l)
    S=0
    i=0
    while S < 0.25 :
        S = S + F[i]
        i = i + 1
    return X[i]

def quartile3(l):
    X,F=evol_distrib_freq(l)
    S=0
    i=0
    while S < 0.75 :
        S = S + F[i]
        i = i + 1
    return X[i]

def action_stat(param,data,cat,sen,*btemp):
    x=select(data,cat,sen,*btemp)
    if param == "minimum":
        return minimum(x)
    if param == "maximum":
        return maximum(x)
    if param == "moyenne":
        return moyenne(x)
    if param == "variance":
        return variance(x)
    if param == "écart-type":
        return sigma(x)
    if param == "médiane":
        return médiane(x)
    if param == "quartile1":
        return quartile1(x)
    if param == "quartile3":
        return quartile3(x)
    
def action_stat2(param,data,cat1,cat2,sen,*btemp):
    x,y=select2(data,cat1,cat2,sen,*btemp)
    if param == "covariance":
        return cov(x,y)
    if param == "corrélation":
        return corrélation(x,y)
#######################################################################
#Mesure de l'indice humidex
#######################################################################
def formule_humidex(temp,hum):
    a,b= 17.27,237.7
    alpha= (a*temp)/(b+temp)+np.log(hum)
    TR=(b*alpha)/(a-alpha)
    return (temp+0.5555*(6.11*np.exp(5417.7530*((1/273.16)-(1/273.15+TR)-10))))
    
def indice_humidex(data,sen,*btemp):
    t,h=select2(data,"temp","humidity",sen,*btemp)
    T,H=np.array(t),np.array(h)
    n = len(T)
    ind=[]
    for i in range(n):
        ind.append(formule_humidex(T[i],H[i]))
    Ind=np.array(ind)
    return Ind

def draw_humidex(data,sen,*btemp):
    Ind=indice_humidex(data,sen,*btemp)
    Time=select(data,"sent_at",sen,*btemp)
    pl.plot(Time,Ind,marker="+",linestyle='None',color='black')
 
#######################################################################
#Mesures de similarité
####################################################################### 
def dtw_method(s,t,fen):
    n, m = len(s), len(t)
    if fen==None:
        fen=max(n,m)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    for i in range(1,n+1):
        for j in range(max(1,i-fen),min(m+1,i+fen+1)):
                       dtw_matrix[i,j]=0
    for i in range(1, n+1):
        for j in range(max(1,i-fen),min(m+1,i+fen+1)):
            c = ((s[i-1] - t[j-1])**2)
            v_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = sqrt(c + v_min)
    return dtw_matrix
#s[0:i] et t[0:j]

def best_path_aux(s,t,fen):
    d = dtw_method(s,t,fen)
    i = len(s)
    j = len(t)
    p = []
    if d[i,j] != -1:
        p.append((i-1,j-1))
    while i>=1 and j>=1:
        r=compmin([d[i-1][j-1],d[i-1][j],d[i][j-1]])
        if r==0:
            i,j=i-1,j-1
        elif r==1:
            i=i-1
        elif r==2:
            j=j-1
        if d[i,j] != -1:
            p.append((i-1,j-1))
    p.pop()
    p.reverse()
    return p

def dtwdistance(data,cat,sen1,sen2,fen,*btemp):
    x,y=select3(data,cat,sen1,sen2,*btemp)
    C=dtw_method(x,y,fen)
    (n,m)=C.shape
    d=C[n-1][m-1]
    return d

def dtwshow(data,cat,sen1,sen2,fen,*btemp):
    s1,s2=select3(data,cat,sen1,sen2,*btemp)
    btempbis=btemp[2:4]
    time1,time2=select(data,"sent_at",sen1,*btemp),select(data,"sent_at",sen2,btempbis)
    plot_warping(s1,s2,time1,time2,fen) 

def dtw_best_path(data,cat,sen1,sen2,fen,*btemp):
    x,y = select3(data,cat,sen1,sen2,*btemp)
    if fen==None:
        fen=max(len(x),len(y))
    return best_path(x,y,fen)

def plot_warping(x,y,time1,time2,fen):
    path=best_path_aux(x,y,fen)
    pl.plot(time1,x,marker="+",linestyle='-',color='red')
    pl.plot(time2,y,marker="+",linestyle='-',color='blue')
    for n, m in path:
        if n < 0 or m < 0:
            continue
        pl.plot([time1[n], time2[m]],[x[n], y[m]],linewidth = 0.5, color = 'green',alpha=0.5)
#######################################################################
#Similarité plus simple
####################################################################### 
def prim_comp(data,cat,sen1,sen2,*btemp):
    x,y=select3(data,cat,sen1,sen2,*btemp)
    k=min(len(x),len(y))
    c=[]
    for i in range(k):
        c.append(abs(x[i]-y[i]))
    return c

def ratio(e,data,cat,sen1,sen2,*btemp): 
    c=prim_comp(data,cat,sen1,sen2,*btemp)
    s=0
    for i in c:
        if i<e:
            s=s+1
    return (s/len(c))

def sim_prim_draw(data,cat,sen1,sen2,*btemp):
    c=np.array(prim_comp(data,cat,sen1,sen2,*btemp))
    X,F = evol_distrib_freq(c)
    pl.plot(X,F,marker="+",linestyle='None')
    pl.axvline(moyenne(c),color='green',linestyle='dashed')
    pl.axvline(maximum(c),color='red',linestyle='-')
    pl.axvline(minimum(c),color='red',linestyle='-')
    pl.axvline(quartile1(c),color='purple',linestyle='dashed')
    pl.axvline(médiane(c),color='purple',linestyle='dashed')
    pl.axvline(quartile3(c),color='purple',linestyle='dashed')
    
def comparaison_primaire(e,data,cat,sen1,sen2,*btemp):
    c=prim_comp(data,cat,sen1,sen2,*btemp)
    n=len(c)
    plage1=extract(data,"sent_at",sen1)
    plage2=extract(data,"sent_at",sen2)
    if btemp == ((),(),(),()) or btemp ==() :
        tinf1=min(plage1)
        tsup1=max(plage1)
        tinf2=min(plage2)
        tsup2=max(plage2)
    else :
        tinf1 = pnd.Timestamp(btemp[0],tz=7200)
        tsup1 = pnd.Timestamp(btemp[1],tz=7200)
        tinf2 = pnd.Timestamp(btemp[2],tz=7200)
        tsup2 = pnd.Timestamp(btemp[3],tz=7200)
    m1=interv(plage1,tinf1,tsup1)[0]
    m2=interv(plage2,tinf2,tsup2)[0]
    T1=np.array(plage1[m1:n])
    T2=np.array(plage2[m2:n])
    S1=np.array((extract(data,cat,sen1))[m1:n])
    S2=np.array((extract(data,cat,sen2))[m2:n])
    pl.plot(T1,S1,marker="+",linestyle='None',color="blue",alpha=0.5)
    pl.plot(T2,S2,marker="+",linestyle='None',color="red",alpha=0.5)
    for i in range(n):
        if c[i]<e:
            pl.plot([T1[i],T2[i]],[S1[i],S2[i]],linewidth = 0.5, color = 'green',linestyle='-')
            pl.plot(T1[i],S1[i],marker='+',color='green')
            pl.plot(T2[i],S2[i],marker='+',color='green')

#######################################################################
#Fonctions auxiliaires
####################################################################### 

def red(l):
    n,M,m=len(l),max(l),min(l)
    if M != m:
        for i in range(n):
            l[i]=(l[i]-m)/(M-m)
        return l
    else :
        for i in range(n):
            l[i]=l[i]/m
        return l


def norm(l):
    s=sigma(l)
    m=moyenne(l)
    n=len(l)
    for i in range(n):
        l[i]=(l[i]-m)/s
    return l

def compmin(a):
    imin, vmin = 0, inf
    for i, v in enumerate(a):
        if v < vmin:
            imin, vmin = i, v
    return imin

def arrange(l1,l2):
    n=len(l1)
    L1=l1
    L1.sort()
    L2=[]
    for i in L1:
        L2.append(l2[l1.index(i)])
    return (np.array(L1),np.array(L2))

def select(data,cat,sen,*btemp):
    plage=extract(data,"sent_at",sen)
    if btemp == ((),()) or btemp ==() :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = pnd.Timestamp(btemp[0],tz=7200)
        tsup = pnd.Timestamp(btemp[1],tz=7200)
    [i,j]=interv(plage,tinf,tsup)
    x=(extract(data,cat,sen))[i:j]
    return x


def select2(data,cat1,cat2,sen,*btemp):
    plage=extract(data,"sent_at",sen)
    if btemp == ((),()) or btemp ==() :
        tinf=min(plage)
        tsup=max(plage)
    else :
        tinf = pnd.Timestamp(btemp[0],tz=7200)
        tsup = pnd.Timestamp(btemp[1],tz=7200)
    [i,j]=interv(plage,tinf,tsup)
    x=(extract(data,cat1,sen))[i:j]
    y=(extract(data,cat2,sen))[i:j]
    return x,y

def select3(data,cat,sen1,sen2,*btemp):
    plage1=extract(data,"sent_at",sen1)
    plage2=extract(data,"sent_at",sen2)
    if btemp == ((),(),(),()) or btemp ==() :
        tinf1=min(plage1)
        tsup1=max(plage1)
        tinf2=min(plage2)
        tsup2=max(plage2)
    else :
        tinf1 = pnd.Timestamp(btemp[0],tz=7200)
        tsup1 = pnd.Timestamp(btemp[1],tz=7200)
        tinf2 = pnd.Timestamp(btemp[2],tz=7200)
        tsup2 = pnd.Timestamp(btemp[3],tz=7200)
    [i,j]=interv(plage1,tinf1,tsup1)
    [e,f]=interv(plage2,tinf2,tsup2)
    s1=np.array((extract(data,cat,sen1))[i:j])
    s2=np.array((extract(data,cat,sen2))[e:f])
    return s1,s2
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
            M.append(np.array(extract(data,j,i)))
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
            X = np.array(extract(data,j,i))
            pl.plot(T,X)
            pl.savefig('testgraph_'+j+'_'+str(i)+'_.pdf',format='pdf')
            pl.cla()
#######################################################################
argums=sys.argv
#######################################################################
#Lecture des variables
#######################################################################
if len(argums)>1:
    action=argums[1]
    if action == "corrélation":
        variable1 = argums[2]
        variable2 = argums[3]
        if len(argums)>4:
            start_date = argums[4]
            end_date = argums[5]
        else:
            start_date,end_date=(),()
    if action == "similarité":
        méthode = argums[2]
        dimension = argums[3]
        capteur1 = int(argums[4])
        capteur2 = int(argums[5])
        if méthode == "primaire":
            marge = float(argums[6])
            if len(argums)>7:
                start_date1 = argums[7]
                end_date1 = argums[8]
                start_date2 = argums[9]
                end_date2 = argums[10]
            else:
                start_date1,end_date1,start_date2,end_date2=(),(),(),()
        elif méthode == "dtw":
            fen = float(argums[6])
            if len(argums)>7:
                start_date1 = argums[7]
                end_date1 = argums[8]
                start_date2 = argums[9]
                end_date2 = argums[10]
            else:
                start_date1,end_date1,start_date2,end_date2=(),(),(),()
    else:
        variable = argums[2]
        if len(argums)>3:
            start_date = argums[3]
            end_date = argums[4]
        else:
            start_date,end_date=(),()
else :
    print("Aucun argument n'a été entré")  
#######################################################################
#Programme principal
#######################################################################
if len(argums)>1:
    if action == "display":
        if variable=="humidex":
             draw_humidex(données,None,start_date,end_date)
             pl.title("Graphique illustrant l'évolution de "+variable+" en fonction du temps entre le "+str(start_date)+" et le "+str(end_date))
             pl.show()
        else:
            fig=pl.figure()
            pl.gcf().subplots_adjust(wspace = 0.5, hspace = 0.5)
            for i in range(1,7):
                fig.add_subplot(4,2,i)
                tracer('black',données,variable,i,start_date,end_date)
                pl.title(variable+" du capteur n°"+str(i)+" entre le "+str(start_date)+" et le "+str(end_date))
            fig.add_subplot(4,2,(7,8))
            tracer('black',données,variable,None,start_date,end_date)
            pl.title("Graphique illustrant l'évolution de "+variable+" en fonction du temps entre le "+str(start_date)+" et le "+str(end_date))
            pl.show()
    elif action == "displayStat":
        fig=pl.figure()
        pl.gcf().subplots_adjust(wspace = 0.5, hspace = 0.5)
        for i in range(1,7):
            fig.add_subplot(4,2,i)
            tracer('black',données,variable,i,start_date,end_date)
            tracer_stat(All,données,variable,i,start_date,end_date)
            pl.title("Stats de "+variable+" du capteur n°"+str(i)+" entre le "+str(start_date)+" et le "+str(end_date))
        fig.add_subplot(4,2,(7,8))
        tracer_stat(All,données,variable,i,start_date,end_date)
        tracer('black',données,variable,None,start_date,end_date)
        pl.title("Graphique illustrant l'évolution de "+variable+" et ses paramètres stats en fonction du temps entre le "+str(start_date)+" et le "+str(end_date))
        pl.show()
    elif action == "corrélation":
        x,y=select2(données,variable1,variable2,None,start_date,end_date)
        C=corrélation(x,y)
        X,Y=np.array(x),np.array(y)
        T=np.array(select(données,"sent_at",None,start_date,end_date))
        pl.plot(T,X,marker="+",linestyle='None',color='red')
        pl.plot(T,Y,marker="+",linestyle='None',color='blue')
        pl.legend(handles=[mp.Patch(label="Coefficient de corrélation : "+str(C))])
        pl.title("Graphique illustrant l'évolution de "+variable1+" et de "+variable2+" et leur indice de corrélation en fonction du temps entre le "+str(start_date)+" et le "+str(end_date))
        print("Valeur du coeff. de corrélation pour le couple ("+variable1+","+variable2+") : "+str(C))
        pl.show()
    elif action == "similarité":
        if méthode=="primaire":
            fig=pl.figure()
            fig.add_subplot(2,2,(1,2))
            comparaison_primaire(marge,données,dimension,capteur1,capteur2,start_date1,end_date1,start_date2,end_date2)
            fig.add_subplot(2,2,(3,4))
            sim_prim_draw(données,dimension,capteur1,capteur2,start_date1,end_date1,start_date2,end_date2)
            print("La similarité selon cette mesure vaut : "+str(ratio(marge,données,dimension,capteur1,capteur2,start_date1,end_date1,start_date2,end_date2)))
        elif méthode == "dtw":
            dtwshow(données,dimension,capteur1,capteur2,fen,start_date1,end_date1,start_date2,end_date2)
            print("La similarité selon cette mesure vaut : "+str(dtwdistance(données,dimension,capteur1,capteur2,fen,start_date1,end_date1,start_date2,end_date2)))
        pl.show()


        
All=["minimum","maximum","moyenne","médiane","quartile1","quartile3","variance","écart-type"]
col=['cyan','red','green','blue','purple','magenta','grey','black']
