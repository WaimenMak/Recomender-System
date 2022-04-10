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
# from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from entities.Movie import Movie

from database.database_connection import SQLiteConnection

def get_recommend_content_based_approach(movies: List[Movie], data, genre_list, user_id, round):

    """
    Summary: 
        - This method is used for the endpoint (POST:/api/recommend)
        - It returns a list of movie recommendations according to the users preferences and the users profile 
        - The method uses a content based approach

    Args: 
        movies: List[Movies] -> List of Movies which the user selected  
        data: the movie input data (movie_info.csv)
        genre_list: list with all possible genres of the movies 


    Returns: 
        results: It returns a list of movie recommendations according to the users preferences. 
        user_profile: the user profile used to create the recommendations


    """
    #The method get_initial_items is called with the highest ranking movie (only one movie is used as input) 
    rec_movies, user_profile = get_initial_items_content_based_approach(movies, data, genre_list, user_id, round)

    #Just the whole data is loaded from the ids 

    #None of these movies have been scored by the user 
    rec_movies.loc[:, 'score'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'poster_url', 'score']]

    return json.loads(results.to_json(orient="records")) ,  json.loads(user_profile.to_json(orient="records"))


def user_add_content_based_approach(movies:List[Movie], user_id, round, algorithm):
    """
    Summary: 
        - This method stores the user's selected movies in a database 
        - The user's preferences are added to the database

    Args: 
        movies: List[Movies] -> List of Movies which the user selected  

    Returns: 
        - void
    

    """
    #Open Connection
    sqlConnection = SQLiteConnection()

    for item in movies: 
        iid = item.movie_id
        score = item.score 
        s = [user_id,str(iid),int(score),'0']
        
        query = f"Insert INTO runtime_u_data VALUES ({user_id}, {str(iid)}, {int(score)}, {int(round)},{int(algorithm)} )"

        sqlConnection.insert_statement(query)

#________________________________Content-based movie recommendation system_________________________________________--
def get_initial_items_content_based_approach(movies:List[Movie], data, genre_list, user_id, round):
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
    user_add_content_based_approach(movies, user_id, round, 1)

    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection

    user_preference_df = pd.read_sql_query(f"SELECT * from runtime_u_data WHERE user_id={user_id}", con)    



    movies_genre_df = data 



    user_preference_df["user_id"] =  user_preference_df["user_id"].astype(int)



    user_preference_df_filtered = user_preference_df[user_preference_df["user_id"]==user_id]
    user_movie_rating_df_filtered = pd.merge(user_preference_df_filtered, movies_genre_df)


    user_movie_df = user_movie_rating_df_filtered.copy(deep=True)
    user_movie_df = user_movie_df[genre_list]

    #Create matrix -> replace all na values
    movies_genre_matrix = movies_genre_df[genre_list].fillna(0).to_numpy()


    rating_weight = user_preference_df_filtered.score / user_preference_df_filtered.score.sum()
    user_profile = user_movie_df.T.dot(rating_weight.values.reshape(rating_weight.shape[0],1))

    user_profile_normalized = user_profile / sum(user_profile.values)


    #Save to database
    user_profile_normalized_sql = user_profile_normalized.copy(deep=True)
    user_profile_normalized_sql= user_profile_normalized_sql.T
    user_profile_normalized_sql["user_id"]= user_id

    updateUserProfile(user_profile_normalized_sql, user_id)
    # user_profile_normalized_sql.to_sql(name='user_profile', con=con, if_exists='append', index=False)

    #Calculate recommendation based on the user profile 
    u_v = user_profile_normalized.to_numpy()

    recommendation_table =  cosine_similarity(u_v.T,movies_genre_matrix)


    recommendation_table_df = movies_genre_df[['movie_id', 'movie_title', 'poster_url']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]

    #Return the top 12 items back 
    res = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[0:12]

    return res, user_profile_normalized_sql
    

def updateUserProfile(user_profile, user_id):
    """
    Summary: 
        - This method inserts the user profile in to the database
        - Only the current version of the user profile is stored. Therefore it overwrites the old version  

    Args: 
        movies: List[Movies] -> List of Movies which the user selected  
        user_id: user's id to whom the user profile belongs

    Returns: 
        - void
    

    """
    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection
    try:
        sql = f'DELETE FROM user_profile WHERE user_id={user_id}'
        cur = con.cursor()
        cur.execute(sql)
        con.commit()
        user_profile.to_sql(name='user_profile', con=con, if_exists='append', index=False)

    
    except Exception: 
        user_profile.to_sql(name='user_profile', con=con, if_exists='append', index=False)




def get_similar_items_content_based_approach(itemid, data, genre_list, user_id):
    """
    Summary: 
        - this method returns similar items to the provided inputed item 

    Args: 
        itemid: Movie for which similar items should be returend 
        data: the movie input data (movie_info.csv)
        genre_list: list with all possible genres of the movies 
        user_id: id of the user to load the user_model  

    Returns: 
        - the top 5 items that are most similar to the provided item 
    
    Comment: 
        Approach 2 is implemented: 
            1. Multiply item genre with current user model 
            2. Calculate cosine similarity towards all other user models 
            3. return the top 5 items 

    """




    #Brainstorming 

    # Approach 1: 
        # 1.Create normal user model based on the current ratings of the user 
        # 2. Filter all the items that are similar to the current item based on genre -> at least the same genres 
        # 3. Join filtered dataset with the results and return the 5 items with the hightes similarity 



        
    # Approach 2: 
        # 1. Multiply item genre with current user model 
        # 2. Calculate cosine similarity towards all other user models 
        # 3. return the top 5 items 

    res = []

    sqlConnection = SQLiteConnection()
    con = sqlConnection.connection



    #Load user profile 
    user_profile = pd.read_sql_query(f"SELECT * from user_profile WHERE user_id={user_id}", con)    



    #Get the item from the dataset
    movies_genre_df = data 
    item = movies_genre_df[movies_genre_df["movie_id"]==int(itemid)]


    #Create matrix -> replace all na values
    movies_genre_matrix = movies_genre_df[genre_list].fillna(0).to_numpy()



    #Calculate recommendation based on the user profile 
    u_v = user_profile.iloc[:,1:].to_numpy()
    u_v = u_v[0].astype(float)
    item_v = item[genre_list].fillna(0).to_numpy()
    # item_v = item.to_numpy()


    user_item_vector = u_v * item_v


    recommendation_table =  cosine_similarity(user_item_vector, movies_genre_matrix)


    recommendation_table_df = movies_genre_df[['movie_id', 'movie_title', 'poster_url']].copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]

    #Filter all items that have the same genres as the provided item 

    #Return the top 12 items back 
    res = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[0:5]

    res.loc[:, 'score'] = None
    results = res.loc[:, ['movie_id', 'movie_title', 'poster_url', 'score']]
    return json.loads(results.to_json(orient="records"))
    