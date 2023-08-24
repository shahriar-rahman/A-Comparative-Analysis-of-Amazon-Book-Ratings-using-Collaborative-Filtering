import pandas as pd
import missingno as msn
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.font_manager import FontProperties
import plotly.express as px
import plotly.graph_objects as go
path_info = '../data/books_data.csv'
path_rating = '../data/Books_rating.csv'


class ConstructFeatures:
    def __init__(self):
        self.df_info = pd.read_csv(path_info)
        self.df_rating = pd.read_csv(path_rating)

    @staticmethod
    def debug_text(title, task):
        print('\n')
        print("=" * 150)
        print('◘ ', title)

        try:
            print(task)

        except Exception as exc:
            print("! ", exc)

        finally:
            print("=" * 150)

    @staticmethod
    def display_dataframe(name, df, contents):
        table = df.head(contents)
        print('\n')
        print("=" * 150)
        print("◘ ", name, " Dataframe:")
        print(table.to_string())
        print("=" * 150)

    def construct_features(self):
        # Inquire structural integrity
        self.debug_text("Statistical Information Data:", self.df_info.describe())
        self.debug_text("General Information Data:", self.df_info.info())

        self.debug_text("Statistical Review Data:", self.df_rating.describe())
        self.debug_text("General Review Data:", self.df_rating.info())

        # Enhance data accessibility
        self.df_info = self.df_info.drop(["image", "previewLink", "publisher", "infoLink"],
                                         axis=1)
        self.df_rating = self.df_rating.drop(["profileName", "review/time", "review/helpfulness", "review/summary"],
                                             axis=1)

        self.df_info.rename(columns={'Title': 'book_title', 'authors': 'book_author', 'publishedDate': 'published_date',
                                     'ratingsCount': 'ratings_count'},
                            inplace=True)
        self.df_rating.rename(columns={'Id': 'book_id', 'Title': 'book_title', 'Price': 'book_price',
                                       'User_id': 'user_id', 'review/text': 'review', 'review/score': 'rating'},
                              inplace=True)

        self.debug_text("Information Data Columns:", self.df_info.columns)
        self.debug_text("Review Data Columns:", self.df_rating.columns)

        # Analyze for any Missing values
        null_checker = self.df_info.isnull().sum()
        self.debug_text("Information Dataframe Null values:", null_checker)

        msn.matrix(self.df_info, color=(0.66, 0.25, 0.013), figsize=[13, 15], fontsize=10)
        plt.title("Missingno Matrix -Information data")
        plt.show()

        null_checker = self.df_rating.isnull().sum()
        self.debug_text("Review Dataframe Null values:", null_checker)

        msn.matrix(self.df_rating, color=(0.66, 0.25, 0.013), figsize=[13, 15], fontsize=10)
        plt.title("Missingno Matrix -Review data")
        plt.show()

        # Clean NaN values from the data
        self.df_info.dropna(axis=0, how='any')
        self.df_rating.dropna(axis=0, how='any')

        # Remove data inconsistency by extracting the year values only
        year = 0
        published_year = []
        exception_list = []

        for sample in self.df_info['published_date']:
            try:
                year = sample.split('-')[0]

            except Exception as exc:
                exception_list.append(exc)
                year = sample

            finally:
                published_year.append(year)

        self.df_info['published_year'] = published_year
        self.df_info = self.df_info.drop(['published_date'], axis=1)
        print("\n! Year value integrated with ", len(exception_list), " Exceptions handled")

        # Dataframe Information
        self.display_dataframe("Information", self.df_info, 20)
        self.display_dataframe("Ratings", self.df_rating, 20)

        return self.df_info, self.df_rating


if __name__ == "__main__":
    main = ConstructFeatures()
    print(main.construct_features())
