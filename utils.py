from sklearn.feature_extraction.text import TfidfVectorizer
from recommendationAlgorithms.processing import preprocessing
import pandas as pd

def map_genre(genre):
    return ""+genre+"==1"


def item_representation_based_book_plots(book_df, max_feat=100):
    tfidf = TfidfVectorizer(preprocessor=preprocessing,
                            ngram_range=(1, 1),
                            max_features=max_feat)
    tfidf_matrix = tfidf.fit_transform(book_df['Book-Title'])

    feature_list = tfidf.get_feature_names()
    book_TF_IDF_vector = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
    book_TF_IDF_vector['itemId'] = book_df['itemId']

    return tfidf_matrix, book_TF_IDF_vector, feature_list


def item_representation_based_movie_plots(movies_df, max_feat=100):
    movies_df['overview'] = movies_df['overview'].fillna('')
    tfidf = TfidfVectorizer(preprocessor=preprocessing,
                            ngram_range=(1, 1),
                            max_features=max_feat)
    tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

    feature_list = tfidf.get_feature_names()
    movie_TF_IDF_vector = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
    movie_TF_IDF_vector['movieId'] = movies_df['movieId']

    return tfidf_matrix, movie_TF_IDF_vector, feature_list