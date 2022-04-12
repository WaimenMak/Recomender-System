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
from database.database_connection import SQLiteConnection
import sys

import simplejson


import recommendationAlgorithms.content_based_recommendation as content_based
from recommendationAlgorithms.item_to_vectore_reommendation import item2vec, item2vec_get_items

from gensim.models import Word2Vec
from scipy.stats import ttest_ind
import json

import os

# Fast API
templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

round = 0
algo_selected = 0

# =======================DATA=========================
data = pd.read_csv("data/movie_info_new.csv")
init_set = set()   # for keywords initial recommendation
model = Word2Vec.load('movies_embedding.model')

# This is the old genre list
# genre_list =["Action", "Adventure", "Animation", "Children", "Comedy", "Crime","Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery","Romance", "Sci_Fi", "Thriller", "War", "Western"]

# This is the new genre list -> for movie_data_new.csv
genre_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

"""
=================== Body =============================
"""

# =======================Website===============================


@app.get("/test", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("/client/index.html", {"request": request})

# == == == == == == == == == API == == == == == == == == == == =


# == == == == == == == == == 1. Get Keywords/ Genres for initial selection
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

# == == == == == == == == == 2. Get Keywords/ Genres for initial selection


@app.post("/api/movies")
def get_movies(firstinput: list):
    global init_set
    keywords = firstinput[0]

    global algo_selected

    # TODO -> implement user_id increment
    # 944 -> 945 -> 946
    # Select (max user_id) FROM database
    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection

    max_user = pd.read_sql_query(
        f"Select max(user_id) as user_id_max FROM runtime_u_data", con)

    if list(max_user["user_id_max"])[0]:
        max_user_id = int(max_user["user_id_max"])
    else:
        max_user_id = 943

    new_user_id = max_user_id + 1
    global user
    user = new_user_id

    algo_selected = firstinput[1]

    print(keywords)
    # _, movie_TF_IDF_vector, _ = item_representation_based_movie_plots(data)
    movie_TF_IDF_vector = pd.read_json("tfidf_mat.json")
    # s = set()
    for kw in keywords:
        for item in movie_TF_IDF_vector[movie_TF_IDF_vector[kw] > 0].movieId:
            init_set.add(item)
    # try:
    print(len(init_set))
    res = np.random.choice(list(init_set), 18, replace=False)
    print(init_set)
    results = data[data['movie_id'].isin(res)]
    print(res)
    results.loc[:, 'score'] = 0

    results = results.loc[:, ['movie_id',
                              'movie_title', 'poster_url', 'score']]

    return json.loads(results.to_json(orient="records"))
    # except:
    #     print(len(init_set))
    #     print("not enough data.")

# == == == == == == == == == 3. Get Recommendation


# This function is trying to fix the problem of duplicated data in content-based method.
def get_right_data(movies: List[Movie]):
    movie_stored = []
    for movie in movies:
        if movie.score > 0:
            movie_stored.append(movie)

    restest = [int(movie.movie_id) for movie in movie_stored]
    rec_movies = data.loc[data['movie_id'].isin(restest)]

    if 'score' not in rec_movies.columns:
        rec_movies.loc[:, 'score'] = 0

    for i in movie_stored:
        rec_movies.score[rec_movies.movie_id == i.movie_id] = i.score

    rec_movies["user_id"] = user
    rec_movies["algorithm"] = algo_selected
    rec_movies["round"] = round

    return rec_movies


@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
    # def get_recommend(movies: List[Movie], algorithm:int, user_id,round ):
    """
    # Summary:
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

    # TODO: at the moment the user id is hardcoded -> should be provided by the function call

    rec_movies = get_right_data(movies)

    # move the part of storing user info from api/profile to here
    update_user_profile_in_database(
        rec_movies.loc[:, ['user_id', 'movie_id', 'score', 'round', 'algorithm']], user)

    print("now round is", round)
    print("type======", type(round))

    # Algo choose in backend cannot work before, because the type of this global property is str !
    if algo_selected == "1":
        # Here the content based algorithm is called
        # recommendations, user_profile = content_based.get_recommend_content_based_approach(movies, data, genre_list, user_id, round)
        recommendations, user_profile = content_based.get_recommend_content_based_approach(
            movies, data, genre_list, user, round)
    else:
        # TODO: implement item-to-factor algorithm
        recommendations = item2vec(
            movies, data, model, user, init_set, 18, round)

    return recommendations


@app.post("/api/record_round")
async def rec_round(round_fronted: list):
    global round
    round = int(round_fronted[0])


# == == == == == == == == == 4. This returns the 5 most simlar items for a given item_id
# TODO: rename to : get_similar_items


# focus
@app.get("/api/get_similar_items/{item_id}")
async def get_similar_items(item_id):
    """
    # Summary:
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

    if algo_selected == "1":
        # Here the content based algorithm is called
        print("algo1--- -> content based approach")
        result = content_based.get_similar_items_content_based_approach(
            item_id, data, genre_list, user_id=user)
    else:
        # TODO: implement item-to-factor algorithm
        print("algo2=== 1")
        result = item2vec_get_items(item_id, data, model)
        print("result number:", result)

    return result


# TODO: -> Refresh must be changed to the new dataset -> else it will not be working
# Each refresh: returns just a new list of moview based on the initial keyword selection
# Always returns 18 movies

# @app.post("/api/refresh")
# def get_movies(genre: list):
#     print("this is refresh", genre)
#     query_str = " or ".join(map(map_genre, genre))
#     results = data.query(query_str)
#     results.loc[:, 'score'] = 0
#     results = results.sample(
#         18).loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'score']]
#
#     print(results)
#
#     return json.loads(results.to_json(orient="records"))

@app.post("/api/refresh")
def get_movies_again(stored_set=init_set):
    try:
        # print(stored_set)
        res = np.random.choice(list(stored_set), 18, replace=True)
        print(res)
        results = data[data['movie_id'].isin(res)]
        results.loc[:, 'score'] = 0
        print(len(results))
        results = results.loc[:, ['movie_id',
                                  'movie_title',  'poster_url', 'score']]
        return json.loads(results.to_json(orient="records"))
    except:
        # print(len(stored_set))
        print("not enough data.")

# try to just store the movies which have rated


@app.post("/api/profile")
def get_profile(movies: List[Movie]):
    movie_stored = []
    for movie in movies:
        if movie.score > 0:
            movie_stored.append(movie)
    if len(movie_stored) == 0:
        return None

    restest = [int(movie.movie_id) for movie in movie_stored]
    rec_movies = data.loc[data['movie_id'].isin(restest)]

    if 'score' not in rec_movies.columns:
        rec_movies.loc[:, 'score'] = 0

    for i in movie_stored:
        rec_movies.score[rec_movies.movie_id == i.movie_id] = i.score
    rec_movies.loc[:, 'rating'] = 0

    # # Set the user_id
    # rec_movies["user_id"]= user
    # rec_movies["algorithm"] = algo_selected
    # rec_movies["round"]= round

    results = rec_movies.loc[:, ['movie_id', 'movie_title',  'poster_url',
                                 'score', 'rating']]

    # TODO: Store the data in a database -> here in the end
    # update_user_profile_in_database(rec_movies.loc[:,['user_id', 'movie_id', 'score', 'round', 'algorithm']], user)

    return json.loads(results.to_json(orient="records"))


def update_user_profile_in_database(movies: List[Movie], user_id: int):
    # 1. Buld delete everything from the old connection
    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection
    try:
        # Get the old user data that is already in the table
        old_data = pd.read_sql_query(
            f"SELECT * from runtime_u_data WHERE user_id={user_id}", con)

        if old_data.shape[0] != 0:
            old_data_movie_id = set(old_data["movie_id"])
            new_data_movie_id = set(movies["movie_id"])
            overlap = old_data_movie_id & new_data_movie_id
            movies[movies["movie_id"].isin(
                overlap)]["movie_id"] = old_data[old_data["movie_id"].isin(overlap)]["round"].values
            sql = f'DELETE FROM runtime_u_data WHERE user_id={user_id}'
            cur = con.cursor()
            cur.execute(sql)

        movies.to_sql(name='runtime_u_data', con=con,
                      if_exists='append', index=False)
        con.commit()

    except:
        # 2. Rollback in case of delete error
        e = sys.exc_info()[0]
        con.rollback()
        print("Error: The update of the user profile did not work in the database")


@app.post("/api/explain")
def get_explaination(movies: List[Movie]):
    index = []
    index.append(
        int(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id))
    results = data.loc[data['movie_id'].isin(index)]

    return json.loads(results.to_json(orient="records"))


@app.get("/api/guesslike/{movie_id}")
async def add_recommend(movie_id):
    print("here")
    print("item2vec get similar items")
    result = item2vec_get_items(movie_id, data, model)
    print("result", result)
    return result
    # print(movie_id)
    # res = get_similar_items_like(str(movie_id), n=6)
    # res = [int(i) for i in res]
    # rec_movies = data.loc[data['movie_id'].isin(res)]
    # rec_movies.loc[:, 'like'] = None
    # results = rec_movies.loc[:, [
    #     'movie_id', 'movie_title', 'poster_url', 'like']]

    # return json.loads(results.to_json(orient="records"))


@app.get("/api/ttest")
async def get_ttest():
    pass
    # 1. Reads the data from the database
    #
    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection

    user_preference_df = pd.read_sql_query(
        f"SELECT * from runtime_u_data", con)

    # 2. Calculate t-Test with the data
    #    2.1 Algo 1 vs. Algo 2
    #    2.2. First Round vs Second Round

    print(user_preference_df)

    algo1_filtered_df = user_preference_df[user_preference_df["algorithm"] == 0]
    algo2_filtered_df = user_preference_df[user_preference_df["algorithm"].isin(set([1,2]))]

    firstround_filtered_df = user_preference_df[user_preference_df["round"] == 1]
    secondround_filtered_df = user_preference_df[user_preference_df["round"] == 1]

    firstround_filtered_df_algo1 = user_preference_df[(user_preference_df["round"] == 1) & (user_preference_df["algorithm"]==0)]
    secondround_filtered_df_algo1 = user_preference_df[(user_preference_df["round"] == 2) &( user_preference_df["algorithm"]==0)]
    firstround_filtered_df_algo2 = user_preference_df[(user_preference_df["round"] == 1) & (user_preference_df["algorithm"]==2)]
    secondround_filtered_df_algo2= user_preference_df[(user_preference_df["round"] == 2 )& (user_preference_df["algorithm"]==2)]

    # 3. Calculate average value per user
    algo1_filtered_df_average = algo1_filtered_df.groupby("user_id")[
        "score"].mean()
    algo2_filtered_df_average = algo2_filtered_df.groupby("user_id")[
        "score"].mean()

    firstround_filtered_df__algo1_average = firstround_filtered_df_algo1.groupby("user_id")[
        "score"].mean()
    secondround_filtered_df__algo1_average = secondround_filtered_df_algo1.groupby("user_id")[
        "score"].mean()


    firstround_filtered_df__algo2_average = firstround_filtered_df_algo2.groupby("user_id")[
        "score"].mean()
    secondround_filtered_df__algo2_average = secondround_filtered_df_algo2.groupby("user_id")[
        "score"].mean()

    t_test_within_algo = ttest_ind(
        algo1_filtered_df_average, algo2_filtered_df_average)
    t_test_within_rounds_algo1 = ttest_ind(
        firstround_filtered_df__algo1_average, secondround_filtered_df__algo1_average)

    t_test_within_rounds_algo2 = ttest_ind(
        firstround_filtered_df__algo2_average, secondround_filtered_df__algo2_average)

    ret = {
        "algo1_results": list(algo1_filtered_df_average.values),
        "algo2_results": list(algo2_filtered_df_average.values),
        "firstround_results_algo1": list(firstround_filtered_df__algo1_average.values),
        "secondround_results_algo1": list(secondround_filtered_df__algo1_average.values),
        "firstround_results_algo2": list(firstround_filtered_df__algo2_average.values),
        "secondround_results_algo2": list(secondround_filtered_df__algo2_average.values),
        "t_test_within_algo_statistic": float(t_test_within_algo[0]),
        "t_test_within_algo_p_value": float(t_test_within_algo[1]),
        "t_test_within_rounds_algo1_statistic": float(t_test_within_rounds_algo1[0]),
        "t_test_within_rounds_algo1_p_value": float(t_test_within_rounds_algo1[1]),
        "t_test_within_rounds_algo2_statistic": float(t_test_within_rounds_algo2[0]),
        "t_test_within_rounds_algo2_p_value": float(t_test_within_rounds_algo2[1]),
    }

    return json.loads(simplejson.dumps(ret, ignore_nan=True))
    # 3. For the visualisation of the result: -> 4 Charts are being displayed ->
    # Return Object should look like this:
    # alg_comparision_algo1: [7.4 ; 4.5; 9.5 ...]
    # alg_comparision_algo2: [7.4 ; 2.5; 5.5 ...]
    # first_round_avg: [4.2, 5.0, ...]
    # second_round_avg: [4.2, 5.0, ...]
    # ttest_within_algo: 0.05
    # ttest_within_round: 0.07

    # Each List in the above object: e.g  [7.4 ; 4.5; 9.5 ...] -> stores all the average ratings of the recommendations for each user

    #
