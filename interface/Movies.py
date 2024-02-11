#!/usr/bin/env python
# coding: utf-8

#    # Import and discover

# In[1]:


import pandas as pd 
import numpy as np 
from collections import Counter
import re 


# In[2]:


df=pd.read_csv("/Users/Lenovo/Desktop/summer/movie_rec/archive (10)/movies_metadata.csv")
df.info()
df.corr()
df


# #                                                Cleaning Phase

# In[3]:


df.columns
df.drop(['belongs_to_collection','homepage','poster_path','tagline','spoken_languages','revenue','budget','imdb_id','video','vote_count','production_companies'],axis=1,inplace=True)


# In[4]:


df.set_index(df['id'],inplace=True)
df.set_index(df.index.astype(str),inplace=True)


# In[5]:


df.loc['82663','title']=df.loc['82663','original_title']
df.loc['2014-01-01','title']=df.loc['2014-01-01','original_title']
df.drop(df[df['title'].isnull()].index,inplace=True)


# In[6]:


def clean_title(title):
    title = re.sub("[^0-9 ]", "", title)
    return title
df['id']=df['id'].astype(str).apply(clean_title)
Counter([i!=1 for i in (Counter(df['id'])).values()])


# In[7]:


df=df.reindex(columns=[ 'title','original_title', 'genres', 'original_language', 'overview',
       'popularity', 'production_countries', 'release_date', 'runtime',
       'status', 'vote_average','adult'])
df.columns


# In[8]:


df['production_countries']=[i[17:19] for i in df['production_countries'].apply(str)]
ll=df[([len(i)<3 for i in df['genres']])].index
df.loc[ll,'genres']="unkown"
df


# In[9]:


df['genres'] = df['genres'].astype(str)
df['genres']=df['genres'].str.split(' ', expand=True)[3]
df['genres']=df['genres'].astype(str).apply(clean_title)
df


# # Search Engine
# 

# In[10]:


from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,5)) #tfidf fun that search for 1 or couple words called vectorizer

tfidf = vectorizer.fit_transform(df["title"]) #fit column re title so all titles have values using tfidf 
dump(vectorizer,"vect.joblib")


# In[11]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title,vectorizer):
    query_vec = vectorizer.transform([title]) #transform the chaine into numbers
    similarity = cosine_similarity(query_vec, tfidf).flatten()#similarity between the query to search nd the fited column tfidf
    indices = np.argpartition(similarity, -5)[-10:]# -5 for the largest 5 values in ordered array by argpartition
    results = df.iloc[indices].iloc[::-1]
    
    return results


# In[12]:


import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

def on_type(data=df):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 1:
            display(search(title))

movie_input.observe(on_type, names='value')


display(movie_input, movie_list)


# # Recommandation sys

# In[15]:


#rating=pd.read_csv("/Users/Lenovo/Desktop/summer/movie_rec/archive (10)/ratings_small.csv")
#rating
#can't be done cause of ID problems


# In[ ]





# In[ ]:




