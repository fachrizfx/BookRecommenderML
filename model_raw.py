import pandas as pd
import numpy as np 
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from google.colab import files
import os
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
!pip install -q kaggle

"""visit: https://www.kaggle.com/docs/api; to see more info about how to use Kaggle's API"""

!kaggle datasets download -d ruchi798/bookcrossing-dataset

local_zip = '/content/bookcrossing-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

books = pd.read_csv('/content/Books Data with Category Language and Summary/Preprocessed_data.csv')
books

books.isnull().sum()

books.drop(['Unnamed: 0', 'img_s', 'img_m', 'img_l', 'Summary', 'location'], inplace=True, axis=1)
books


books.drop(index=range(500000, 1031175), inplace=True, axis=0)
books

# Univariate Data Analysis

books.info()

books.hist()

books.isnull().sum()

print('jumlah data: ', len(books))
print('skala rating dari {0} sampai {1}'.format(books['rating'].min(), books['rating'].max()))
print('banyak kategori buku: ', len(books['Category'].unique()))
print('macam-macam bahasa dalam buku: ', books['Language'].unique())

books[books['Language']=='9']


# Data Preparation

books = books.sort_values('isbn', ascending=True)
books

# Data Cleaning

books.isnull().sum()

books = books.replace(to_replace='9', value=np.nan)
books[books['Language']=='9'].sum()

len(books[books['Language']=='9'])
books.isnull().sum()

books = books.dropna()

books.isnull().sum()

print('after data cleaning dataset size:', len(books))

len(books['Category'].unique())

preparation = books
preparation = preparation.drop_duplicates('isbn')
preparation

# Feature Selection

references = ['isbn', 'book_title', 'book_author', 'publisher', 'Language', 'Category']

book_isbn = preparation[references[0]].tolist()
 
book_title = preparation[references[1]].tolist()
 
book_author = preparation[references[2]].tolist()

book_publisher = preparation[references[3]].tolist()

book_lang = preparation[references[4]].tolist()

book_category = preparation[references[5]].tolist()
 
print(len(book_isbn))
print(len(book_title))
print(len(book_author))
print(len(book_publisher))
print(len(book_lang))
print(len(book_category))

books_new = pd.DataFrame({
    'isbn': book_isbn,
    'title': book_title,
    'author': book_author,
    'publisher': book_publisher,
    'language': book_lang,
    'category': book_category
})
books_new

"""# Modelling"""

data = books_new

vec = CountVectorizer()
 
vec.fit(data['category']) 
 
vec.get_feature_names()

vec_matrix = vec.fit_transform(data['category']) 
 
vec_matrix.shape

vec_matrix.todense()

# Cosine Similarity

cosine_sim = cosine_similarity(vec_matrix) 
cosine_sim

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['title'], columns=data['title'])
print('Shape:', cosine_sim_df.shape)
 
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

def get_recommendations(book_title, similarity_data=cosine_sim_df, items=data[['title', 'category']], k=5):
    index = similarity_data.loc[:,book_title].to_numpy().argpartition(
        range(-1, -k, -1))
    
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    closest = closest.drop(book_title, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

# Evaluation

books.sample(10)

books[books['book_title']=='Dragonshadow']

get_recommendations('Dragonshadow', k=10)

def get_precision(relev, sum):
    percentage = print('Recommendation System Precision Percentage: {0}%'.format((relev / sum * 100)))
    return percentage

get_precision(10, 10)

get_recommendations('Under the Tuscan Sun: At Home in Italy', k=20)

get_precision(20, 20)
