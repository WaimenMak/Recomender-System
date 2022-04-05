# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 22:11
# @Author  : Weiming Mai
# @FileName: algo2.py
# @Software: PyCharm

# from content_based_recommendation import user_add_content_based_approach
import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from processing import preprocessing
import csv
import json
from content_based_recommendation import user_add_content_based_approach

def user_add(iid, score):
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter='\t')
        data_input = []
        s = [user,str(iid),int(score),'0']
        data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def store_result(store_list, mid, title, exp):
  entry = {
      "movieId": int(mid),
      "title": title,
      "liked": None,
      "explaination": exp
  }
  store_list.append(entry)
  return store_list

def item2vec(movies, data, model, user_id, init_set, n, round):
    iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    if round > 1:
        finetune_model(movies, model)

    # iid = str(sorted(movies, key=lambda i: i['score'], reverse=True)[0]['movieId'])
    # score = int(sorted(movies, key=lambda i: i['score'], reverse=True)[0]['score'])

    # user_add(iid, score)
    user_add_content_based_approach(movies, user_id, round)
    # s = set()
    ls = []
    for movie in movies:
      # if movie.score >= 4:
        if movie.score >= 4:
        # sim = model.most_similar([str(movie.movieId)], topn=10)
            sim = model.most_similar([str(movie.movie_id)], topn=10)
            for item in sim:
                title = data.loc[data['movie_id']==int(item[0]),'title'].values[0]
                exp = f"Your interested movie: {movie.title} has {item[1]:.2f} similarity to movie: {title}"
                # s.add(item[0])
                store_result(ls, item[0], title, exp)
                  # recommendation = temp.loc[temp['movieId']==item[0]]
    m = len(ls)
    print(m)
    if m > n:
        # res = np.random.choice(list(s), n)
        res = random.sample(ls, n)
        results = pd.DataFrame(res)
        return json.loads(results.to_json(orient="records"))
    elif m < n and m > 0:
        results = pd.DataFrame(ls)
        return json.loads(results.to_json(orient="records"))
    elif m == 0:
        res = np.random.choice(list(init_set), n)
        res = [int(i) for i in res]
        rec_movies = data.loc[data['movie_id'].isin(res)]
        print(rec_movies)
        rec_movies.loc[:, 'like'] = None
        rec_movies.loc[:, 'explaination'] = "Choosen from your keywords"
        results = rec_movies.loc[:, ['movie_id', 'title', 'like', "explaination"]]
        return json.loads(results.to_json(orient="records"))


def finetune_model(movies, model):
    interested = []
    not_interested = []

    for movie in movies:
        if movie.score >= 4:
            interested.append(str(movie.movie_id))
        else:
            not_interested.append(str(movie.movie_id))
    if len(interested) > 0 or len(not_interested) > 0:
        new_sentense = [interested, not_interested]
        model.train(new_sentense, total_examples=model.corpus_count, epochs=model.epochs)

def get_similar_items(iid, data, model):
    res = model.most_similar([str(iid)], topn=5)
    res = [int(item[0]) for item in res]
    rec_movies = data.loc[data['movieId'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    rec_movies.loc[:, 'explaination'] = "5 most similar movies"
    results = rec_movies.loc[:, ['movieId', 'title', 'like', "explaination"]]
    return json.loads(results.to_json(orient="records"))

@app.post("/api/refresh")
def refresh_movies():
    """
    refresh the movies after clicking the refresh button
    :return: 
    """
    res = np.random.choice(list(init_set), 18)
    results = data[data['movieId'].isin(res)]
    print(results)
    return json.loads(results.to_json(orient="records"))
