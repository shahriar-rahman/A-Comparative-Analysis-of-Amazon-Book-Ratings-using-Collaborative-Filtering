import os
import sys
import numpy as np
import pandas as pd
import missingno as msn
from pandas.plotting import table
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.font_manager import FontProperties
import plotly.express as px
import plotly.graph_objects as go
path_info = '../../data/books_data.csv'
path_rating = '../../data/Books_rating.csv'
sys.path.append(os.path.abspath('../visualization'))
import visualize


class ConstructFeatures:
    def __init__(self):
        self.df_info = pd.read_csv(path_info)
        self.df_rating = pd.read_csv(path_rating)
        self.visualize = visualize.Visualize()

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

    @staticmethod
    def save_df(command, name):
        try:
            command.to_excel(f"../../records/{name}.xlsx")

        except Exception as exc:
            print("! ", exc)

        plot = plt.subplot(111, frame_on=False)
        plot.xaxis.set_visible(False)
        plot.yaxis.set_visible(False)
        table(plot, command, loc='upper right')
        plt.savefig(f"../../records/{name}.png")

    def construct_features(self):
        # Inquire structural integrity
        self.debug_text("• Statistical Info:", self.df_info.describe())
        self.debug_text("• General Info:", '')
        self.df_info.info()

        self.debug_text("• Statistical Rating:", self.df_rating.describe())
        self.debug_text("• General Rating:", '')
        self.df_rating.info()

        self.save_df(self.df_info.describe(), "raw_info_describe")
        self.save_df(self.df_rating.describe(), "raw_rating_describe")

        # Enhance data accessibility
        self.debug_text("• Raw Info Columns:", self.df_info.columns)
        self.debug_text("• Raw Rating Columns:", self.df_rating.columns)

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

        self.debug_text("• Modified Info Columns:", self.df_info.columns)
        self.debug_text("• Modified Rating Columns:", self.df_rating.columns)

        # Analyze for any Missing values
        msn.matrix(self.df_info, color=(0.18, 0.32, 0.17), figsize=[17, 20], fontsize=10)
        plt.title("Missingno Matrix for Info", fontsize=15, fontweight='bold')
        plt.show()

        null_checker = self.df_info.isnull().sum()
        self.debug_text("Information Dataframe Null values:", null_checker)

        msn.matrix(self.df_rating, color=(0.18, 0.32, 0.17), figsize=[17, 20], fontsize=10)
        plt.title("Missingno Matrix for Rating", fontsize=15, fontweight='bold')
        plt.show()

        null_checker = self.df_rating.isnull().sum()
        self.debug_text("Review Dataframe Null values:", null_checker)

        # Data cleaning
        self.df_info.dropna(axis=0, how='any')
        self.df_rating.dropna(axis=0, how='any')

        # Find duplicates and perform the De-duplication process
        duplicated_cells = 0
        check_duplicate = self.df_rating.duplicated()

        for row in check_duplicate:
            if row:
                duplicated_cells += 1

        duplicated_prc = (duplicated_cells / len(check_duplicate)) * 100
        self.debug_text("• Total Cells:", len(check_duplicate))
        self.debug_text("• Duplicated Cells:", duplicated_cells)
        self.debug_text("• Duplicate %:", duplicated_prc)

        data_values = np.array([duplicated_cells, len(self.df_rating)-duplicated_cells])
        explode = [0.2, 0]
        labels = ["Duplicate Cells", "Unique Cells"]
        title = "Pie Chart: Unique vs Duplicate cells (Pre-processing)"
        self.visualize.plot_pie(data_values, explode, labels, title)

        self.df_rating = self.df_rating.drop_duplicates(subset=None, keep='first', inplace=False,
                                                        ignore_index=False)

        # Validate de-duplication
        duplicated_cells = 0
        check_duplicate = self.df_rating.duplicated()

        for row in check_duplicate:
            if row:
                duplicated_cells += 1

        self.debug_text("• Number of duplicated cells after de-duplication:", duplicated_cells)

        data_values = np.array([duplicated_cells, len(self.df_rating) - duplicated_cells])
        explode = [0.2, 0]
        labels = ["Duplicate Cells", "Unique Cells"]
        title = "Pie Chart: Unique vs Duplicate cells (Post-processing)"
        self.visualize.plot_pie(data_values, explode, labels, title)

        # Maintain Feature consistency
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

        # Review Dataframe
        self.display_dataframe("Information", self.df_info, 20)
        self.display_dataframe("Ratings", self.df_rating, 20)

        self.debug_text("• Modified Statistical Info:", self.df_info.describe())
        self.debug_text("• Modified General Info:", '')
        self.df_info.info()

        self.debug_text("• Modified Statistical Rating:", self.df_rating.describe())
        self.debug_text("• Modified General Rating:", '')
        self.df_rating.info()

        self.save_df(self.df_info.describe(), "mod_info_describe")
        self.save_df(self.df_rating.describe(), "mod_rating_describe")

        print('-' * 150)
        return self.df_info, self.df_rating


if __name__ == "__main__":
    main = ConstructFeatures()
    print(main.construct_features())
