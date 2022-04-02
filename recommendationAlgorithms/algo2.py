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

def item2vec(books: List[book]):
    iid = str(sorted(books, key=lambda i: i.score, reverse=True)[0].movie_id)
    # score = int(sorted(books, key=lambda i: i.score, reverse=True)[0].score)





