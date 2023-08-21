import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy
from nltk.sentiment import SentimentIntensityAnalyzer
from surprise import accuracy
import sys
import os
sys.path.append(os.path.abspath('../'))
from py_utils import generic_utils


class AmazonRecommendation:
    def __init__(self):
        self.gu = generic_utils.GenericUtils()
        self.df_book_ratings = pd.read_csv('../../data/Books_rating.csv')

    def insight_analysis(self):

        # Feature Structuring
        self.df_book_ratings.rename(columns={'Id': 'book_id', 'User_id': 'user_id', 'review/text': 'review',
                                             'Title': 'title', 'review/score': 'rating'}, inplace=True)
        self.df_book_ratings = self.df_book_ratings[['book_id', 'user_id', 'review', 'title', 'rating']]

        # Dataframe Diagnosis
        self.gu.view_dataframe(self.df_book_ratings, 20)
        n_books = self.gu.df_length(self.df_book_ratings, 'book_id')
        n_users = self.gu.df_length(self.df_book_ratings, 'user_id')
        print(f"\nNumber of Books: {n_books}  \nNumber of Unique Users: {n_users}")

        # Extraction Substantial Features
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


if __name__ == "__main__":
    main = AmazonRecommendation()
    main.insight_analysis()

