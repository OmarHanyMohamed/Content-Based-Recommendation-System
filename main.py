# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:12:05 2021

@author: omar
"""

from reco_model import get_recommendation, genre_recommendation
from flask import Flask, jsonify, request
import tmdbsimple as tmdb

app = Flask(__name__)
tmdb.API_KEY = 'API_KEY'

@app.route('/', methods=['GET'])
def home():
    return "Welcome to RecoMovies"

@app.route('/api', methods=['Post'])
def recommend():
    # Request movie input as json 
    movie = request.get_json(force=True)
    movie_name = movie['name'] # get attribute name which contain name of movie
    results = get_recommendations(movie_name) # call function from reco_model
    data = []
    search = tmdb.Search()
    
    # for each title in results get it's information 
    for title in results.title.values.tolist():
        response = search.movie(query=title)
        
        # [0] to only get first output 
        # different movies can have the same title
        data.append(search.results[0]) 

    return jsonify({"results": data})


@app.route('/genre', methods=['Post'])
def genre():
    # Request genre from the user
    genre = request.get_json(force=True)
    genre_name = genre['type'] # get attribute type which contain genre the user wants
    results = genre_recommendation(genre_name) # call function from reco_model
    data = []
    search = tmdb.Search()

    # same as recommend
    for title in results:
        response = search.movie(query=title)
        data.append(search.results[0])
        
    return jsonify({ 'results': data })

if __name__ == '__main__':
    # 0.0.0.0 simply to run flask on your local IP address
    app.run(debug=True, host= '0.0.0.0') 
