# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 22:11
# @Author  : Weiming Mai
# @FileName: algo2.py
# @Software: PyCharm

# from content_based_recommendation import user_add_content_based_approach
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from processing import preprocessing
import csv

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


# def item_representation_based_book_plots(book_df, max_feat=100):
#     tfidf = TfidfVectorizer(preprocessor=preprocessing,
#                             ngram_range=(1, 1),
#                             max_features=max_feat)
#     tfidf_matrix = tfidf.fit_transform(book_df['Book-Title'])
#
#     feature_list = tfidf.get_feature_names()
#     movie_TF_IDF_vector = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
#     movie_TF_IDF_vector['itemId'] = book_df['itemId']
#
#     return tfidf_matrix, movie_TF_IDF_vector, feature_list

def item_representation_based_book_plots(book_df, max_feat=100):
    tfidf = TfidfVectorizer(preprocessor=preprocessing,
                            ngram_range=(1, 1),
                            max_features=max_feat)
    tfidf_matrix = tfidf.fit_transform(book_df['Book-Title'])

    feature_list = tfidf.get_feature_names()
    movie_TF_IDF_vector = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
    movie_TF_IDF_vector['itemId'] = book_df['itemId']

    return tfidf_matrix, movie_TF_IDF_vector, feature_list
