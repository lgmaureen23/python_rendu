#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:04:21 2020

@author: legermaureen
"""

import pandas as pd
import csv 
import matplotlib.pyplot as plt
import numpy as np 

with open("EIVP_KM.csv")as f :  
    tableau=[]
    lire=csv.reader(f)
    for ligne in lire:
        tableau.append(ligne[0].split(';'))

# tableau = [['id;noise;temp;humidity;lum;co2;send_at],[],...]
#len(tableau)=7881
        
#détermination de l'indicatif de la colonne 
def indicatif(var):
    liste=['id','noise','temp','humidity','lum','co2','send_at']
    for i in range(len(liste)):
        if var==liste[i]:
            return i
      
#récupérer une colonne
def colonne(tab,num):
    col=[]
    n=len(tab)
    for i in range(n): 
        col.append(tab[i][num]) 
    return col
#on a ['send_at','...','...',...] pour colonne(tableau,6)

#chaque colonne désigne une valeur qui est une str 
#il faudra faire attention à convertir en int pour chaque uso 
ident = colonne(tableau,0)[1::]
noise = colonne(tableau,1)[1::]
temperature= colonne(tableau,2)[1::]
humidity = colonne(tableau,3)[1::]
luminosity = colonne(tableau,4)[1::]
co2=colonne(tableau,5)[1::]
sendat =colonne(tableau,6)[1::]

#SEPARATION DES CAPTEURS 

#détermination des index de chaque capeteurs 
def lim_capteur(num_capteur):
    n=len(ident)
    lim=0
    for i in range(n):
        if int(ident[i])==num_capteur:
            lim+=1  #inclus 
    a='num_capteur'
    return ident.index(a),lim+1 #exclu

def differencier_capteur(num_capteur):
    debut,fin=lim_capteur(num_capteur)
    return noise[debut,fin], temperature[debut,fin],temperature[debut,fin],humidity[debut,fin],luminosity[debut,fin],co2[debut,fin],sendat[debut,fin]
#ce sont toujours des listes de chaines de caracteres 




#NOTION DE TEMPS 
#Extraction de la date
def extr_date(chaine):
    return chaine[0:19]

#extr_date('2019-08-25 11:45:54 +0200')
#Out[57]: '2019-08-25 11:45:54'

#Indice d'intervalle limite 
#y=colonne(tableau,6)
def lim_time(y,start_date,end_date):
    #récupération des index 
    d,f=y.index('start_date'),y.index('end_date')
    return d,f
#value ERROR 'start_date is not in list  

# Calcul de valeurs statistiques

#Calcul du minimum
def min (liste):
    minim=liste[0]
    for elem in liste:
        if elem<minim:
            minim=elem
    return minim
    
#Calcul du maximum
def max (liste):
    maxi=liste[0]
    for elem in liste:
        if elem>maxi:
            maxi=elem
    return maxi

#Calcul des moyennes

def moyenne_arithmetique(liste):
    n=len(liste)
    S=0
    for elem in liste:
        S+=elem
    return S/n

def moyenne_geometrique(liste):
    n=len(liste)
    P=1
    for elem in liste:
        P=P*elem
    return P**(1/n)

#Calcul de la variance

def variance(liste):
    moy=moyenne_arithmetique(liste)
    n=len(liste)
    S=0
    for elem in liste:
        S+=(elem-moy)**2
    return S/n
    
#Calcul de l'écart type
 
def ecart_type(liste):
    return variance(liste)**(1/2)
    
#Calcul de la médiane (on créé d'abord une fonction de tri pour trier la fonction d'entrée, on va appliquer ici le tri par tri-fusion, sinon on peut utiliser la fonction sorted(liste))

def insere(x,liste):

    if liste==[]:
        return [x]
    elif x<=liste[0]:
        return [x] + liste
    else:
        return [liste[0]] + insere(x,liste[1:len(liste)])

def fusion(liste1,liste2):

    if liste1==[]:
        return liste2
    elif liste2==[]:
        return liste1
    else:
        return fusion(liste1[1:len(liste1)],insere(liste1[0],liste2))

def tri_fusion(liste):

    n=len(liste)

    if n==0 or n==1:
        return liste
    else:
        return fusion(tri_fusion(liste[0:n//2]),tri_fusion(liste[n//2:n]))

#Calcul de la médiane

def mediane (liste):
    liste_triee=tri_fusion(liste)
    n=len(liste)
    if n%2==0:
        return (liste_triee[(n)//2]+liste_triee[(n+2)//2])/2   
    else:
        return liste_triee[(n-1)//2]

#Fonctions auxquelles on doit faire appel 
def tracer(data,cat,sen,tinf,tsup):
    None
    
#On part du principe qu'on a la liste des valaurs sous forme de flaots correspondant à un capteur et à un paramètre    
#Mise en place des calculs statistiques, calcul point par point 

def display_stats(liste,temps,cat):
    moyarithcal,moygeocal,variancecal,ecart_typecal,medianecal=[],[],[],[],[]
    n=len(liste)
    res=[liste[0]]
    for i in range(1,n):  
        #calcul des données statistiques pour les 2ers termes jusqu'à l'ensemble de tous les termes 
        res.append(liste[i])
        moyarithcal.append(moyenne_arithmetique(res))
        moygeocal.append(moyenne_geometrique(res))
        variancecal.append(variance(res))
        ecart_typecal.append(ecart_type(res))
        medianecal.append(mediane(res))
        # pour le tracé au fur et à mesure 
        tracer(moyarithcal,cat,temps[0],temps[i])
        tracer(moygeocal,cat,temps[0],temps[i])
        tracer(variancecal,cat,temps[0],temps[i])
        tracer(ecart_typecal,cat,temps[0],temps[i])
        tracer(medianecal,cat,temps[0],temps[i])
        