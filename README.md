# Content-Based-Recommendation-System
Implementing a content based recommendation system using **cosine similarity**, which recommends movies similar to the movie user likes.

## Description

I used IMDB's weighted rating formula to construct my chart. Mathematically, it is represented as follows:

<img src="https://farm3.static.flickr.com/2384/2137446131_2a4e10d9a5.jpg" width="500"/>

I built an engine that computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked. Since we will be using movie metadata (or content) to build this engine, this also known as Content Based Filtering.

I built two Content Based Recommenders based on:

* Movie Overviews and Taglines
* Movie Cast, Crew, Keywords and Genre

### How Cosine Similarity works?

Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.

<img src="https://assets.datacamp.com/production/repositories/4966/datasets/ec0fa4795484baf3a19c3f345755a85457a2aae4/cosine.png" width="500"/>

## About the Data 
* I had to use the small dataset (due to the computing power I possess being very limited)
* We have **9099** movies avaiable in our small movies metadata dataset which is 5 times smaller than our original dataset of **45000** movies.

## Built with 
* Python 3.7.2
* Flask 1.1.2

## Notes
* This dataset is missing one file which is credits.csv as it's large so you can download it and the entire data from [here](https://www.kaggle.com/rounakbanik/the-movies-dataset) .
* I implemented this recommendation system as a flask api and used it in my app [Recomovie](https://github.com/OmarHanyMohamed/RecoMovieApp) App.
* You can find all required libraries needed in ```requirements.txt``` just run this command ```pip install -r requirements.txt```
