import os
import sys
import pandas as pd
import joblib as jb
from surprise import NMF, SVD
from surprise import accuracy
from surprise import KNNWithMeans
from surprise import Dataset, Reader
from nltk.sentiment import SentimentIntensityAnalyzer
from surprise.model_selection import train_test_split
sys.path.append(os.path.abspath('../'))
from py_utils import generic_utils
from py_utils import ml_utils
split_test = 0.25


class NMFTrain:
    def __init__(self):
        self.current_model = 'NMF'
        self.data_type = 'base'
        self.gu = generic_utils.GenericUtils()
        self.ml = ml_utils.MlUtils()
        self.df_book_ratings = pd.read_csv('../../data_set/Books_rating.csv')

    def data_wrangling(self):
        # Feature Structuring
        self.df_book_ratings.rename(columns={'Id': 'book_id', 'User_id': 'user_id', 'review/text': 'review',
                                             'Title': 'title', 'review/score': 'rating'}, inplace=True)
        self.df_book_ratings = self.df_book_ratings[['book_id', 'user_id', 'review', 'title', 'rating']]

        # Dataframe Diagnosis
        self.gu.view_dataframe(self.df_book_ratings, 20)
        n_books = self.gu.df_length(self.df_book_ratings, 'book_id')
        n_users = self.gu.df_length(self.df_book_ratings, 'user_id')
        print(f"\nNumber of Books: {n_books}  \nNumber of Unique Users: {n_users}")

        # Extraction of Substantial Features
        users_review = self.df_book_ratings.groupby('user_id').count()['review'] > 200
        substantial_users = users_review[users_review].index
        filtered_users = self.df_book_ratings[self.df_book_ratings['user_id'].isin(substantial_users)]

        books_review = filtered_users.groupby('title').count()['review'] >= 50
        substantial_books = books_review[books_review].index
        self.df_book_ratings = filtered_users[filtered_users['title'].isin(substantial_books)]

        self.df_book_ratings = self.df_book_ratings.reset_index()
        print("\n• Substantial Features:")
        self.gu.view_dataframe(self.df_book_ratings, 20)
        print("\nShape of the Dataframe with substantial features: ", self.df_book_ratings.shape)

    def train_model(self):
        # Model Parameters Extraction
        path = '../../data_set/test_data.csv'
        columns = ['user_id', 'book_id', 'rating']

        train, test = self.ml.partition_data(self.df_book_ratings, columns, split_test)
        self.ml.construct_model(self.current_model, train, self.data_type)
        self.gu.save_dataframe(test, columns, path)


if __name__ == "__main__":
    main = NMFTrain()
    main.data_wrangling()
    main.train_model()
