import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import math

from numpy.linalg import svd
from numpy.linalg import svd
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import *
from sklearn.model_selection import train_test_split

import warnings
from sklearn.decomposition import TruncatedSVD


from numpy import cos, sin, arcsin, sqrt
from math import radians

import math

def distance(lat1, lon1,lat2, lon2 ):

    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def Sort(sub_li): 
  

    return(sorted(sub_li, key = lambda x: x[1])) 


def SVD(M,dim):
    Mt=np.transpose(M)
    
    
    
    prd=np.dot(M,Mt)
    
    
    eigenvalue,eigenvec=np.linalg.eig(prd)
    
    
    sortindex=eigenvalue.argsort()[::-1]
    
    

    eigenvalue=eigenvalue[sortindex]    
    
    
    U=eigenvec[:,sortindex]
    U=U[:,0:dim]
    U=np.real(U)
    U=np.around(U,decimals=2)
  
     
    sigma=np.sqrt(abs(eigenvalue))
    sigma=sigma[0:dim]
    sigma=np.around(sigma,decimals=2)
    

    prd=np.dot(Mt,M)
    eigenvalue,eigenvec=np.linalg.eig(prd)
    sortindex=eigenvalue.argsort()[::-1]
    V=eigenvec[:,sortindex]
    V=V[:,0:dim]
    V=np.real(V)
    V=np.around(V,decimals=2) 
    
    
    return U,sigma,V


def ran():
    y=random.randint(0,5)
    if(y<=4):
        y=0
    else:
        y=random.randint(1,5)
    return y  


def main():
    da=pd.read_csv('new.csv')
    
    m=211
    n=1000
    ma=[[0]*m for i in range(n)]
    for i in range(n):
        for j in range(m):
            ma[i][j]=ran()
   
    r=pd.DataFrame(ma)

    latitudee=da['latitude'].tolist()
    longitude=da['longitude'].tolist()
    places=da['Places'].tolist()
    

    user_means = np.array(r.mean(axis = 1)).reshape(-1, 1)
    svd_places = r.div(r.mean(axis = 1), axis = 0)
    svd_places_matrix = svd_places.as_matrix()


    U, sigma, Vt = SVD(svd_places_matrix, 10)

    Vt=Vt.T

    sigma = np.diag(sigma)

    predicted = np.dot(np.dot(U, sigma), Vt)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) * user_means


    predicted_df = pd.DataFrame(predicted_ratings, columns= r.columns)

    
    return places,latitudee,longitude,r,predicted_df,svd_places
    
def already(user):
    al=[]
    for i in range(211):
        if(r[i][user]!=0):
            al.append(places[i])
    return al       
    

def svd_recommender(df_predict, user, umr, number_recomm,unrated):
    user_predicted_places = df_predict.loc[user, :].sort_values(ascending = False)
    original_data = umr.loc[user, :].sort_values(ascending = False)

    
    recommendations = df_predict.loc[user][unrated]
    
    recommendations = pd.DataFrame(recommendations.sort_values(ascending = False).index[:number_recomm])
    
    return recommendations


def final_places(user):
    places,latitudee,longitude,r,predicted_df,svd_places=main()
    ur=[]
    for i in range(211):
        if(r[i][1]==0):
            ur.append(i)
    recommend =svd_recommender(predicted_df, 1, svd_places, 20,ur)
    y=recommend[0][0]
    Recommend=[]

    for i in range(20):
        Recommend.append(places[recommend[0][i]])
    like=y

    har=[]
    ansf=[]
    for i in range(20):
        if(i!=0):    
            j=distance(latitudee[like],longitude[like],latitudee[recommend[0][i]],longitude[recommend[0][i]])
            har.append([recommend[0][i],j])
    

    har=Sort(har)
    
    ansf.append(places[like])
    for i in range(19):
        j=har[i][0]
        ansf.append(places[j]) 
        
    return ansf


h=final_places(0)
