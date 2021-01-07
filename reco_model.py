# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 03:18:11 2020

@author: Omar
"""

import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer

import warnings;

warnings.simplefilter('ignore')

# Reading the data to pandas dataframe
meta = pd.read_csv('./datasets/movies_metadata.csv')

# As there maybe more than one genre per movie so we'll use contain them in lists
# Filling NA values with empty list
meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

vote_counts = meta[meta['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = meta[meta['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

# Number of minimum votes
m = vote_counts.quantile(0.95)

# Creating new feature year from release date
meta['year'] = pd.to_datetime(meta['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# Now let's create the dataframe of the top movies
qualified = meta[(meta['vote_count'] >= m) & (meta['vote_count'].notnull()) & (meta['vote_average'].notnull())][
    ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
# print(qualified.shape)


# Therefore, to qualify to be considered for the chart, a movie has to have at least **434 votes** on TMDB.
# We also see that the avg rating for a movie on TMDB is 5.244 on a scale of 10. 2274 Movies qualify to be on our chart.


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

# We see that three Christopher Nolan Films, Inception, The Dark Knight & Interstellar occur at the top of our chart.
# The chart also indicates a strong bias of TMDB Users towards particular genres and directors.
# Let us now construct our function that builds charts for particular genres.
# For this, we will use relax our default conditions to the **85th** percentile instead of 95.
s = meta.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = meta.drop('genres', axis=1).join(s)


def genre_recommendation(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(20)

    return qualified.title.values


#print(genre_recommendation('Action'))

# Content Based Recommender
# The recommender we built in the previous section suffers some severe limitations.
# For one, it gives the same recommendation to everyone, regardless of the user's personal taste.
# If a person who loves romantic movies (and hates action) were to look at our Top 15 Chart, s/he wouldn't probably like most of the movies.
# If s/he were to go one step further and look at our charts by genre, s/he wouldn't still be getting the best recommendations.

links_small = pd.read_csv('./datasets/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

meta = meta.drop([19730, 29503, 35587])

meta['id'] = meta['id'].astype('int')

smd = meta[meta['id'].isin(links_small)]

smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

# Cosine Similarity

# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies.
# Mathematically, it is defined as follows:

# $cosine(x,y) = \frac{x. y^\intercal}{||x||.||y||} $

# Since we have used TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score.
# Therefore, we will use sklearn's **linear_kernel** instead of cosine_similarities since it is much faster.

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# We now have a pairwise cosine similarity matrix for all the movies in our dataset.
# The next step is to write a function that returns the 30 most similar movies based on the cosine similarity score.
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# We're all set. Let's now try & get the top recommendations for a few movies and see how good the recommendations are.

credits = pd.read_csv('./datasets/credits.csv', engine='python')
keywords = pd.read_csv('./datasets/keywords.csv')

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')

meta['id'] = meta['id'].astype('int')
meta = meta.merge(credits, on='id')
meta = meta.merge(keywords, on='id')

smd = meta[meta['id'].isin(links_small)]
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x, x, x])

# We will do a small amount of pre-processing of our keywords before putting them to any use.
# As a first step, we calculate the frequenct counts of every keyword that appears in the dataset.
s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()

# Keywords occur in frequencies ranging from 1 to 610. 
# We do not have any use for keywords that occur only once. 
# Therefore, these can be safely removed. Finally, we will convert every word to its stem 
# So that words such as Dogs and Dog are considered the same.
 
s = s[s > 1]

stemmer = SnowballStemmer('english')
stemmer.stem('dogs')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# we will add a mechanism to remove bad movies and return movies which are popular and have had a good critical response.
# I will take the top 25 movies based on similarity scores and calculate the vote of the 60th percentile movie. 
# Then, using this as the value of m , we will calculate the weighted rating of each movie using IMDB's formula.

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


print(get_recommendations('Inception'))
