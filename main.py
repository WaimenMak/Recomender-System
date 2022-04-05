from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset
from entities.Movie import Movie
from utils import item_representation_based_movie_plots


import  recommendationAlgorithms.content_based_recommendation as content_based
from  recommendationAlgorithms.item_to_vectore_reommendation import item2vec, get_similar_items

from gensim.models import Word2Vec

import os

## Fast API 
templates = Jinja2Templates(directory="templates")     
     
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================DATA=========================
data = pd.read_csv("data/movie_info_new.csv")
init_set = set()   # for keywords initial recommendation
model = Word2Vec.load('movies_embedding.model')

#This is the old genre list
# genre_list =["Action", "Adventure", "Animation", "Children", "Comedy", "Crime","Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery","Romance", "Sci_Fi", "Thriller", "War", "Western"]

#This is the new genre list -> for movie_data_new.csv
genre_list=['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

"""
=================== Body =============================
"""

#=======================Website===============================
@app.get("/test", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("/client/index.html",{"request": request}) 

# == == == == == == == == == API == == == == == == == == == == =



#== == == == == == == == == 1. Get Keywords/ Genres for initial selection  
# show four genres
# @app.get("/api/genre")
# def get_genre():
#     return {'genre': ["Action", "Adventure", "Animation", "Children"]}


@app.get("/api/genre")
def get_genre():
    return {'genre': ["child", "escape", "family", "friend"]}

# show all generes
'''
@app.get("/api/genre")
def get_genre():
    return {'genre': ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                      "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery",
                      "Romance", "Sci_Fi", "Thriller", "War", "Western"]}
    return{'genre': ['Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']}
'''

#== == == == == == == == == 2. Get Keywords/ Genres for initial selection  
# @app.post("/api/movies")
# def get_movies(genre: list):
#     print(genre)
#     query_str = " or ".join(map(map_genre, genre))
#     results = data.query(query_str)
#     results.loc[:, 'score'] = None
#     results = results.sample(18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
#     return json.loads(results.to_json(orient="records"))

@app.post("/api/movies")
def get_movies(keywords: list):
    print(keywords)
    # _, movie_TF_IDF_vector, _ = item_representation_based_movie_plots(data)
    movie_TF_IDF_vector = pd.read_json("tfidf_mat.json")
    # s = set()
    for kw in keywords:
        for item in movie_TF_IDF_vector[movie_TF_IDF_vector[kw]>0].movieId:
            init_set.add(item)
    res = np.random.choice(list(init_set), 18)
    results = data[data['movie_Id'].isin(res)]
    print(results)
    return json.loads(results.to_json(orient="records"))
#== == == == == == == == == 3. Get Recommendation
@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
# def get_recommend(movies: List[Movie], algorithm:int, user_id,round ):
    """
    ### Summary: 
    - this function is called each time a new recommendation is made -> as input a set of movies with ratings is provided 
    - depending on the specified algorithm either Algorithm 1 or Algorithm 2 is chose for the computation of the recommendation 
        - Algorithm 1: content-based algorithm with cosine similarity 
        - Algorithm 2: item to factor algorithm 

    Args:
        movies (List[Movie]): List of movies with ratings 
        algorithm: Which algorithm should be used to execute the function -> possible values: 0,1 
        user_id: id of the user that made the request 
        round: the recommendation round 


    Returns:
        - Algorithm 1: 
            1. recommendationList: list with movie recommendations for the user 
            2. userProfile: user profile for the user 
        - Algorithm 2: 
            1. recommendationList: list with movie recommendations for the user 
            2. similarity Score:
        
    """     

    #TODO: at the moment the user id is hardcoded -> should be provided by the function call
    algorithm =2
    if algorithm==1: 
        #Here the content based algorithm is called 
        # recommendations, user_profile = content_based.get_recommend_content_based_approach(movies, data, genre_list, user_id, round)
        recommendations, user_profile = content_based.get_recommend_content_based_approach(movies, data, genre_list, 944, 1)
    else: 
        #TODO: implement item-to-factor algorithm
        recommendations = item2vec(movies, data, model, 944, init_set, 18, 1)

    return recommendations

#== == == == == == == == == 4. This returns the 5 most simlar items for a given item_id 
#TODO: rename to : get_similar_items
@app.get("/api/get_similar_items/{item_id}")
async def get_similar_items(item_id,algorithm, user_id):
    """
    ### Summary: 
    - this function is called to retrieve the 5 most similar items 
    - depending on the specified algorithm either Algorithm 1 or Algorithm 2 is chose for the computation of the recommendation 
        - Algorithm 1: content-based algorithm with cosine similarity 
        - Algorithm 2: item to factor algorithm 

    Args:
        item_id: item id for which the 5 most similar items should be retrieved 
        algorithm: Which algorithm should be used to execute the function -> possible values: 0,1 
        user_id: id of the user that made the request 


    Returns:
        - Algorithm 1: 
            1. similarityList: 5 most similar items for a given algorithm 
        - Algorithm 2: 
            1. similarityList: 5 most similar items for a given algorithm 
        
    """     
    algorithm = 1
    if algorithm==1: 
        #Here the content based algorithm is called 
        result = content_based.get_similar_items_content_based_approach(item_id, data, genre_list, user_id=944)
    else: 
        #TODO: implement item-to-factor algorithm
        result = get_similar_items(item_id, data, model)

    return result



#== == == == == == == == == 5. Update the already rated items 
@app.post("/api/update_recommend/{item_id}")
async def update_recommend(item_id, algorithm: int, round: int):
    pass


#== == == == == == == == == 6. Remove the already rated items 
@app.delete("/api/delete_recommend/{item_id}")
async def update_recommend(item_id, algorithm: int, round: int ):
    pass
    # 1. Remove the entry from the database 
    # 2. Recalculate the recommendation list and return to the user 


