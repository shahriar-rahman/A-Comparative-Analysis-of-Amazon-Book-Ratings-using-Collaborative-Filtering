import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNWithMeans
from surprise.model_selection import train_test_split
from surprise import accuracy
from nltk.sentiment import SentimentIntensityAnalyzer
from surprise import accuracy


class AmazonRecommendation:
    def __init__(self):
        self.df_book_ratings = pd.read_csv('../data/Books_rating.csv')

    def exploratory_analysis(self):
        # Integrity Observation
        print("• Book Ratings Dataframe:")
        self.df_book_ratings.info()

        print("\n• Dataframe Structural information:")
        self.df_book_ratings.describe()

        print("\n• Dataframe Sample:")
        df_table = self.df_book_ratings.head(10)
        print(df_table.to_string())

        # Feature Structuring
        self.df_book_ratings.rename(columns={'Id': 'book_id', 'User_id': 'user_id', 'review/text': 'review',
                                             'Title': 'title', 'review/score': 'rating'}, inplace=True)
        self.df_book_ratings = self.df_book_ratings[['book_id', 'user_id', 'review', 'title', 'rating']]

        print("\n• Dataframe after filtering features:")
        df_table = self.df_book_ratings.head(10)
        print(df_table.to_string())

        n_books = len(self.df_book_ratings['book_id'])
        n_users = len(self.df_book_ratings['user_id'])
        print(f"\nNumber of Unique Books: {n_books}  \nNumber of Unique Users: {n_users}")

        # Extracting Substantial Features
        users_review = self.df_book_ratings.groupby('user_id').count()['review'] > 200
        substantial_users = users_review[users_review].index
        filtered_users = self.df_book_ratings[self.df_book_ratings['user_id'].isin(substantial_users)]

        books_review = filtered_users.groupby('title').count()['review'] >= 50
        substantial_books = books_review[books_review].index
        self.df_book_ratings = filtered_users[filtered_users['title'].isin(substantial_books)]

        df_table = self.df_book_ratings.head(10)
        print("\n• Substantial Features:")
        print(df_table.to_string())

        self.df_book_ratings = self.df_book_ratings.reset_index()
        print("\nShape of the Dataframe with substantial features: ", self.df_book_ratings.shape)

    @staticmethod
    def construct_model(df):
        # Load data from a file
        reader = Reader(line_format='user item rating', sep=',')
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader=reader)

        # Split the data into training and testing sets
        train_set, test_set = train_test_split(data, test_size=0.25)

        raw_ratings = data.raw_ratings
        print("\n• Raw Values of the modified Dataframe:")
        print(raw_ratings[:5])

        return train_set, test_set

    def train_model(self):
        # train, test = self.construct_model(self.df_book_ratings)
        self.df_book_ratings = self.df_book_ratings[['user_id', 'book_id', 'rating']]

        # Load data from a file  rating_scale=(1, 5), skip_lines=1
        reader = Reader(line_format='user item rating', sep=',')

        df_table = self.df_book_ratings.head(10)
        print(df_table.to_string())

        data = Dataset.load_from_df(self.df_book_ratings[['user_id', 'book_id', 'rating']], reader=reader)

        # Split the data into training and testing sets
        train, test = train_test_split(data, test_size=0.25)

        raw_ratings = data.raw_ratings
        print("\n• Raw Values of the modified Dataframe:")
        print(raw_ratings[:5])

        # Train a KNN With Means model on the combined score
        model = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
        model.fit(train)

        # Evaluate the model on the test set
        predictions = model.test(test)
        rmse = accuracy.rmse(predictions)
        print("\n• RMSE:", rmse)
        print(predictions[:5])


if __name__ == "__main__":
    amazon = AmazonRecommendation()
    amazon.exploratory_analysis()
    amazon.train_model()
