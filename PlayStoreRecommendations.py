#!/usr/bin/env python
# coding: utf-8

# Project reference: https://www.analyticsvidhya.com/blog/2020/08/recommendation-system-k-nearest-neighbors/#80e4
# 
# Dataset: https://www.kaggle.com/datasets/lava18/google-play-store-apps?resource=download
# 
# GitHub Repo: https://github.com/kiergc/Play-Store-Recommendation-System

# In[518]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[519]:


try:
    googleplaystore_data = pd.read_csv('./googleplaystore.csv')
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)


# # Cleaning and Wrangling

# In[520]:


googleplaystore_data.drop(10472, inplace=True)
googleplaystore_data = googleplaystore_data.drop(columns=['Size','Price','Current Ver'])
googleplaystore_data = googleplaystore_data.drop_duplicates(subset='App',keep="first")


# ### Popularity (Combination of Ratings and Reviews)

# In[521]:


googleplaystore_data = googleplaystore_data.assign(Popularity=lambda row: (row['Rating']-3.5)*(row['Reviews'].astype(int)))
googleplaystore_data['Popularity'] = googleplaystore_data['Popularity'].fillna(0)
googleplaystore_data.head()


# ### Genres
# 
# Create and refine a list of all the unique genres.

# In[522]:


googleplaystore_data['Genres'] = googleplaystore_data['Genres'].str.split(";")
genres_lst = googleplaystore_data['Genres'].explode().unique()
remove_lst = ['Action & Adventure','Educational','Music & Video','Music & Audio']
genres_lst = sorted(list(set(genres_lst)-set(remove_lst)) + ['Audio','Video'])
len(genres_lst), genres_lst


# One-hot encode the genres for each app and store in a column called "Genres Bins".

# In[523]:


def one_hot_genres(row_genres):
    one_hot = []
    
    for genre in genres_lst:
        
        #special cases
        if (genre == 'Action' or genre == 'Adventure') and 'Action & Adventure' in row_genres:
            one_hot.append(1)
            continue
        elif genre == 'Audio' and 'Music & Audio' in row_genres:
            one_hot.append(1)
            continue
        elif genre == 'Video' and 'Music & Video' in row_genres:
            one_hot.append(1)
            continue
        elif genre == 'Music' and ('Music & Audio' in row_genres or 'Music & Video' in row_genres):
            one_hot.append(1)
            continue
        elif genre == 'Education' and 'Educational' in row_genres:
            one_hot.append(1)
            continue
            
        if genre in row_genres:
            one_hot.append(1)
        else:
            one_hot.append(0)
    
    
    return one_hot


# In[524]:


googleplaystore_data['Genres Bins'] = googleplaystore_data['Genres'].apply(lambda x: one_hot_genres(x))
googleplaystore_data.head()


# ### Keywords

# In[525]:


from wordcloud import STOPWORDS
from string import punctuation
from nltk.stem.snowball import SnowballStemmer

stop_words = STOPWORDS
stemmer = SnowballStemmer("english")
punc = punctuation+'™'

def get_keywords(app_name):
    app_name = app_name.lower().translate({ord(i): None for i in punc}).split(' ')
    return [stemmer.stem(word) for word in app_name if word not in stop_words and word != '']


# In[526]:


googleplaystore_data['Keywords'] = googleplaystore_data['App'].apply(lambda x: get_keywords(x))
googleplaystore_data.head()


# In[527]:


from collections import Counter
kword_lst = list(googleplaystore_data['Keywords'].explode())
kword_lst_counter = Counter(kword_lst)
kword_lst = sorted([str(k) for k,v in kword_lst_counter.items() if v > 1])
kword_lst


# In[528]:


def one_hot_kwords(row_kwords):
    one_hot = []
    
    for kword in kword_lst:
        if kword in row_kwords:
            one_hot.append(1)
        else:
            one_hot.append(0)
            
    return one_hot


# In[529]:


googleplaystore_data['Keywords Bins'] = googleplaystore_data['Keywords'].apply(lambda x: one_hot_kwords(x))
googleplaystore_data.head()


# # Implement KNN Algorithm
# 
# TODO: Implement filters

# In[534]:


from scipy import spatial

def calcSimilarity(app1id, app2id):
    app1 = googleplaystore_data.loc[app1id]
    app2 = googleplaystore_data.loc[app2id]
    
    return spatial.distance.cosine(app1['Genres Bins'],app2['Genres Bins']) + spatial.distance.cosine(app1['Keywords Bins'],app2['Keywords Bins'])

def getNeighbors(app, K=10, filters=[]):
    same_category = googleplaystore_data[googleplaystore_data['Category']==app['Category']]
    same_category = same_category.drop(app.name)
    same_category = same_category.drop_duplicates(subset='App',keep="first")

    similarities = []
    for index, curr_app in same_category.iterrows():
        similarities += [calcSimilarity(app.name, index)]
    
    same_category.insert(9,'Similarity',similarities)
    same_category = same_category.sort_values(by=['Similarity','Popularity'],ascending=[True,False])
    
    return same_category.iloc[:K]


# In[542]:


def recommend_ui():
    name = input("\nEnter an app name: ")
    apps = googleplaystore_data[googleplaystore_data['App'].apply(lambda x: x.lower()).str.contains(name.lower())]
    while len(apps)==0:
        print("\nNo app with that name was found. Please try again.\n")
        recommend_ui()
    if len(apps) > 1:
        print("\nMultiple apps with that name were found.")
        print(apps[['App']])
        appid = int(input("\nPlease enter the index of the app you wish to select. "))
    else:
        appid = apps.index[0]
    base_app = googleplaystore_data.loc[appid]
    print("\nSelected App: ", base_app['App'])
    okay = input("\nIf this is correct, press enter. ")
    if okay != '':
        recommend_ui()
        return

    K = input("\n\nHow many recommendations would you like? ")
    try:
        K = int(K)
    except Exception as e:
        K = 10
    if K>20:
        K = 20
        
    #ask here for filters
    
    neighbors = getNeighbors(base_app,K)
    print("\n\nRecommended Apps:\n")
    for index, neighbor in neighbors.iterrows():
        print(neighbor['App'])
        print("Genres: "+str(neighbor['Genres']).strip("[]").replace("'","")+" | Rating: "+str(neighbor['Rating'])+"\n")
    return


# In[543]:


recommend_ui()

