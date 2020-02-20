#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import random as rd
from math import sqrt


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/machine-learning-course/syllabus/gh-pages/hiw-2019b/dataset-students-ml-2019b.csv')
df['nim'] = df['nim'].astype(str)


# In[3]:


#delete unused column
df.drop(['gender','birth_city','birth_date','height','weight','religion'],axis=1,inplace=True)
change = {"dow_of_birth": {"Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6, "Sunday":7}}
df.replace(change,inplace=True)


# In[4]:


def centroid():
    key = ("1","2","3","4")
    value = []
    for x in range(4):
        value.append(rd.randrange(0,65,1))
    return dict(zip(key,value))

centroid()


# In[5]:


def euclidean(centroid,key,data):
#     print(type(df.loc[centroid[key]]['dow_of_birth']))
#     print(type(data['dow_of_birth']))
    
    dow_birth = (df.loc[centroid[key]]['dow_of_birth'] - data['dow_of_birth'])**2
    shoulder = (df.loc[centroid[key]]['shoulder_width'] - data['shoulder_width'])**2
    shirt = (df.loc[centroid[key]]['shirt_height'] - data['shirt_height'])**2

    return sqrt(dow_birth+shoulder+shirt)


# In[6]:


def Coordinate_Mean(lst):
    
    lst[0] /= lst[3] if lst[3] > 0 else 0
    lst[1] /= lst[3] if lst[3] > 0 else 0
    lst[2] /= lst[3] if lst[3] > 0 else 0
    
    return [lst[0],lst[1],lst[2]]


# In[7]:


def NewCoordinate(kmeans_langkah1,data):
    #cN = [dow_birth,shoulder,shirt,counter]
    c1 = [0,0,0,0]
    c2 = [0,0,0,0]
    c3 = [0,0,0,0]
    c4 = [0,0,0,0]
    
    
    for x in range(len(kmeans_langkah1)):
        a = kmeans_langkah1[x] 
        if a['cluster'] == '1':
            c1[0] += data[data['nim'] == a['nim']]['dow_of_birth'].values[0]
            c1[1] += data[data['nim'] == a['nim']]['shoulder_width'].values[0]
            c1[2] += data[data['nim'] == a['nim']]['shirt_height'].values[0]
            c1[3] += 1
        elif a['cluster'] == '2':
            c2[0] += data[data['nim'] == a['nim']]['dow_of_birth'].values[0]
            c2[1] += data[data['nim'] == a['nim']]['shoulder_width'].values[0]
            c2[2] += data[data['nim'] == a['nim']]['shirt_height'].values[0]
            c2[3] += 1
        elif a['cluster'] == '3':
            c3[0] += data[data['nim'] == a['nim']]['dow_of_birth'].values[0]
            c3[1] += data[data['nim'] == a['nim']]['shoulder_width'].values[0]
            c3[2] += data[data['nim'] == a['nim']]['shirt_height'].values[0]
            c3[3] += 1
        elif a['cluster'] == '4':
            c4[0] += data[data['nim'] == a['nim']]['dow_of_birth'].values[0]
            c4[1] += data[data['nim'] == a['nim']]['shoulder_width'].values[0]
            c4[2] += data[data['nim'] == a['nim']]['shirt_height'].values[0]
            c4[3] += 1
    
    return {"1": Coordinate_Mean(c1),"2": Coordinate_Mean(c2),"3": Coordinate_Mean(c3),"4": Coordinate_Mean(c4)}
    


# In[8]:


def euclidean2(koordinat,data): #koordinat di loop buat cluster nya. koordinat[0]=c1, koordinat[1]=c2 ,... dst
    
    dow_birth = (koordinat[0] - data['dow_of_birth'])**2
    shoulder = (koordinat[1] - data['shoulder_width'])**2
    shirt = (koordinat[2] - data['shirt_height'])**2

    return sqrt(dow_birth+shoulder+shirt)


# In[9]:


def Kmeans_1(df,ct):   #masih belom bikin langkah ke-3
    kunci = ('nim','cluster','distance')
    temp_kmeans = []
    
    for idx in range(len(df)):
        min1 = 999999
        nim = ''
        ctr = ''
        temp_euclid = []

        for key in ct:
            hasil = euclidean(ct,key,df.loc[idx])
            if hasil < min1:
                min1 = hasil
                nim = df.loc[idx]['nim']
                ctr = key

        temp_euclid.append(nim)
        temp_euclid.append(ctr)        
        temp_euclid.append(min1)

        a = dict(zip(kunci,temp_euclid))

        temp_kmeans.append(a)
            
    return temp_kmeans
                


# In[10]:


def Kmeans_2(df,new_c):
    kunci = ('nim','cluster','distance')
    temp_kmeans = []
    
    for idx in range(len(df)):
        min1 = 999999
        nim = ''
        ctr = ''
        temp_euclid = []

        for key in new_c:
            hasil = euclidean2(new_c[key],df.loc[idx])
            if hasil < min1:
                min1 = hasil
                nim = df.loc[idx]['nim']
                ctr = key

        temp_euclid.append(nim)
        temp_euclid.append(ctr)        
        temp_euclid.append(min1)

        a = dict(zip(kunci,temp_euclid))

        temp_kmeans.append(a)
            
    return temp_kmeans


# In[11]:


def Kmeans_main(df):
    ct1 = centroid()
    hasil1 = Kmeans_1(df,ct1)
#     print(hasil1)
    prev_ct = {}
    ct2 = NewCoordinate(hasil1,df)
#     print(ct2)
    
    while prev_ct!=ct2:
        prev_ct = ct2
        hasil2 = Kmeans_2(df,ct2)
        ct2 = NewCoordinate(hasil2,df)
    
    print('Model Code: E')
    for x in hasil2:
        print('%s,%s,%0.4f' % (x['nim'], x['cluster'], x['distance']))


# In[12]:


a = Kmeans_main(df)

