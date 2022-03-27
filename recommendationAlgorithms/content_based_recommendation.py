from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from sklearn.metrics.pairwise import cosine_similarity
from entities.Movie import Movie

from database.database_connection import SQLiteConnection

def get_recommend_content_based_approach(movies: List[Movie], data, genre_list, user_id):

    """
    Summary: 
        - This method is used for the endpoint (POST:/api/recommend)
        - It returns a list of movie recommendations according to the users preferences. 
        - The method uses a content based approach

    Args: 
        movies: List[Movies] -> List of Movies which the user selected  
        data: the movie input data (movie_info.csv)
        genre_list: list with all possible genres of the movies 


    Returns: 
        It returns a list of movie recommendations according to the users preferences. 

    """
    #The method get_initial_items is called with the highest ranking movie (only one movie is used as input) 
    rec_movies = get_initial_items_content_based_approach(movies, data, genre_list, user_id)

    #Just the whole data is loaded from the ids 

    #None of these movies have been liked by the user 
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


def user_add_content_based_approach(movies:List[Movie], user_id):
    """
    Summary: 
        - This method stores the user's selected movies in a .csv file 
        - The file with user recommendations is copied and during runtime a separate file is used. 
        - The user's preferences are added to this separate file 

    Args: 
        movies: List[Movies] -> List of Movies which the user selected  

    Returns: 
        - void
    

    """
    #-> get new user id for user -> must be unique for session
    #Test: just one id which is out of scope
    # user_id=944

    # # simulate adding a new user into the original data file
    # #here the original data file ./u.data is copied to new_u.data 
    # df = pd.read_csv('./u.data', sep="\t")
    # df.to_csv('new_' + 'u.data',sep="\t",  index=False)
    
    # with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
    #     #A new line with the user's prefered rating is added to the database 
    #     wf = csv.writer(cfa,delimiter='\t')
    #     data_input = []

    #     #Here all items of the initial recommendation are added to the database
    #     for item in movies: 
    #         iid = item.movie_id
    #         score = item.score 
    #         s = [user_id,str(iid),int(score),'0']
    #         data_input.append(s)
    #         for k in data_input:
    #             wf.writerow(k)

    #Open Connection
    sqlConnection = SQLiteConnection()

    for item in movies: 
        iid = item.movie_id
        score = item.score 
        s = [user_id,str(iid),int(score),'0']
        
        query = f"Insert INTO runtime_u_data VALUES ({user_id}, {str(iid)}, {int(score)}, '0')"

        sqlConnection.insert_statement(query)

#________________________________Content-based movie recommendation system_________________________________________--
def get_initial_items_content_based_approach(movies:List[Movie], data, genre_list, user_id):
    """
    Summary: 
        - This method adds the user's preferences to the databasea by calling user_add_content_based_approach()
        - The it creates movie recommendations based on the initial preferences of the user
        - The user's preferences are added to this separate file 

    Args: 
        movies: List[Movies] -> List of Movies which the user selected  
        data: the movie input data (movie_info.csv)
        genre_list: list with all possible genres of the movies 

    Returns: 
        - a list of movie recommendations according to the user's initial preferences 
    
    Comment: 
        - The content-based approach has the problem that only movies are recommended that are already in the genre of the user 
        - in the beginning it could therefore be useful to show as many genres as possible -> to have ratings for each genere and not only one genre 

    """
    res = []
    user_add_content_based_approach(movies, user_id)

    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection

    user_preference_df = pd.read_sql_query(f"SELECT * from runtime_u_data WHERE user_id={user_id}", con)    



    movies_genre_df = data 



    user_preference_df["user_id"] =  user_preference_df["user_id"].astype(int)



    user_preference_df_filtered = user_preference_df[user_preference_df["user_id"]==user_id]
    user_movie_rating_df_filtered = pd.merge(user_preference_df_filtered, movies_genre_df)

    # user_movie_rating_df_filtered = user_movie_rating_df[user_movie_rating_df["user_id"]==9940]



    user_movie_df = user_movie_rating_df_filtered.copy(deep=True)
    user_movie_df = user_movie_df[genre_list]

    #Create matrix -> replace all na values
    movies_genre_matrix = movies_genre_df[genre_list].fillna(0).to_numpy()


    rating_weight = user_preference_df_filtered.rating / user_preference_df_filtered.rating.sum()
    user_profile = user_movie_df.T.dot(rating_weight.values.reshape(rating_weight.shape[0],1))

    user_profile_normalized = user_profile / sum(user_profile.values)


    #Save to database
    user_profile_normalized_sql = user_profile_normalized.copy(deep=True)
    user_profile_normalized_sql["user_id"]= "user_id"
    user_profile_normalized_sql.to_sql(name='user_profile', con=con, if_exists='append')

    #Calculate recommendation based on the user profile 
    u_v = user_profile_normalized.to_numpy()

    recommendation_table =  cosine_similarity(u_v.T,movies_genre_matrix)


    recommendation_table_df = movies_genre_df[['movie_id', 'movie_title', 'release_date', 'poster_url']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]

    #Return the top 12 items back 
    res = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[0:12]

    return res
    