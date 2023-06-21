import pandas as pd
import missingno as msn
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.font_manager import FontProperties
import plotly.express as px
import plotly.graph_objects as go
path_info = '../data/books_data.csv'
path_rating = '../data/Books_rating.csv'


class FeatureAnalysis:
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

        print("=" * 150)

    @staticmethod
    def display_dataframe(name, df, contents):
        table = df.head(contents)
        print('\n')
        print("=" * 150)
        print("◘ ", name, " Dataframe:")
        print(table.to_string())
        print("=" * 150)

    def data_engineering(self):
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

    def data_inspection(self):
        # Inspect distribution for Ratings
        bins = 10
        subtext = "Book Ratings"
        x_label = "Ratings"
        y_label = "Frequency"

        kind = 'hist'
        self.plot_dataframe(self.df_rating['rating'], subtext, kind, x_label, y_label, "Histogram")

        kind = 'kde'
        # self.plot_dataframe(self.df_rating['rating'], subtext, kind, x_label, y_label, "KDE")

        # Acquire top 10 Book Genres
        subtext = "Book Genres"
        genre = self.df_info['categories'].value_counts().sort_values(ascending=False)
        genre = genre.head(10)
        self.debug_text(subtext, genre)

        # Inspect distribution for Genres
        explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        labels = genre.keys().map(str)
        self.plot_pie(genre, subtext, labels, explode)

        kind = 'barh'
        x_label = "Frequency"
        y_label = "Genres"
        self.plot_dataframe(genre, subtext, kind, x_label, y_label, "Bar")

        # Identify how much Book Ratings affect its Prices
        ratings_price = self.df_rating[['rating', 'book_price']]
        ratings_price = ratings_price.sort_values(by=['book_price', 'rating'], ascending=False)
        self.display_dataframe("Ratings vs Price", ratings_price, 20)

        x = self.df_rating['rating']
        y = self.df_rating['book_price']

        subtext = "Ratings and Price"
        x_label = 'Ratings'
        y_label = 'Book Price'
        self.plot_scatter(x, y, subtext, x_label, y_label)

        # Books most purchased by users
        most_purchases = self.df_rating.groupby('book_title')['user_id'].count().sort_values()
        df_temp_rating = most_purchases.to_frame()
        df_temp_rating['most_purchases'] = most_purchases
        df_arg = df_temp_rating['most_purchases'].sort_values(ascending=False)
        self.display_dataframe("Books most purchased", df_arg, 15)

        # Inspect query from the visualization chart
        x = most_purchases.values[-15:]
        y = most_purchases.index[-15:]
        subtext = "Books most bought by users"
        x_label = "Purchases"
        y_label = "Books"
        self.plot_plotly_bar(x, y, subtext, x_label, y_label)

        # Highest Rated Books (Mean)
        highest_rated = self.df_rating.groupby('book_title')['rating'].mean()
        df_temp_rating = highest_rated.to_frame()
        df_temp_rating['mean_ratings'] = highest_rated
        df_arg = df_temp_rating['mean_ratings'].sort_values(ascending=False)
        self.display_dataframe("Highest Rated Books", df_arg, 15)

        # Inspect query from the visualization chart
        x = highest_rated.values[-15:]
        y = highest_rated.index[-15:]
        subtext = "Books with the highest Rating"
        x_label = "Ratings"
        y_label = "Books"
        self.plot_plotly_bar(x, y, subtext, x_label, y_label)

        # Expensive Books in store (highest mean Price)
        expensive_books = self.df_rating.groupby('book_title')['book_price'].mean()
        df_temp_rating = expensive_books.to_frame()
        df_temp_rating['mean_price'] = expensive_books

        df_arg = df_temp_rating['mean_price'].sort_values(ascending=False)
        self.display_dataframe('Top Expensive Books', df_arg, 15)

        # Distribution of Mean Book Prices
        subtext = "Book Prices"
        x_label = "Price Range ($)"
        y_label = "Frequency"
        self.plot_dataframe(df_arg, subtext, 'hist', x_label, y_label, "Histogram")

        # Top-rated Books accumulating over 3500 Ratings in total (per book)
        accumulated_ratings = self.df_info[self.df_info['ratings_count'] > 3500][['book_title', 'ratings_count']]\
            .drop_duplicates()
        subtext = "Books over 3500 Ratings"
        df_arg = accumulated_ratings.sort_values(by=['ratings_count'], ascending=False)
        self.display_dataframe(subtext, df_arg, 15)

        # Generate a Bar Plot for visual evidence
        x = accumulated_ratings['ratings_count']
        y = accumulated_ratings['book_title']
        x_label = "Ratings"
        y_label = "Books"
        self.plot_plotly_bar(x, y, subtext, x_label, y_label)

        # Total books in a particular category
        category_books = self.df_info.groupby('categories')['book_title'].count().sort_values()
        df_temp_info = category_books.to_frame()
        df_temp_info['category_books'] = category_books

        df_arg = category_books.sort_values(ascending=False)
        subtext = "15 Top Books in a Category"
        self.display_dataframe(subtext, df_arg, 15)

        # Inspect query from the visualization chart
        x = category_books.values[-15:]
        y = category_books.index[-15:]
        x_label = "Books"
        y_label = "Categories"
        self.plot_plotly_bar(x, y, subtext, x_label, y_label)

        # Authors with the most published books
        author_publish = self.df_info.groupby('book_author')['book_title'].count().sort_values().sort_values()
        df_temp_info = author_publish.to_frame()
        df_temp_info['author_publish'] = author_publish

        df_arg = author_publish.sort_values(ascending=False)
        subtext = "Most Books by Author"
        self.display_dataframe(subtext, df_arg, 15)

        # Represent the query via a Bar chart
        x = author_publish.values[-15:]
        y = author_publish.index[-15:]
        x_label = "Authors"
        y_label = "Publishes"
        self.plot_plotly_bar(x, y, subtext, x_label, y_label)

        # Author's active years
        author_years = self.df_info.groupby('book_author')['published_year'].nunique()
        df_temp_info = author_years.to_frame()
        df_temp_info['author_years'] = author_years

        subtext = "Years most active by Authors"
        df_arg = author_years.sort_values(ascending=False)
        self.display_dataframe(subtext, df_arg, 15)

        # Inspect the distribution for the  most active years
        subtext = "Top 15 Active Authors"
        df_arg = df_arg.head(15)
        explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        labels = df_arg.keys().map(str)
        self.plot_pie(df_arg, subtext, labels, explode)

        # Authors who worked in a variety genres of book genres
        author_categories = self.df_info.groupby('book_author')['categories'].nunique()
        df_temp_info = author_categories.to_frame()
        df_temp_info['author_categories'] = author_categories

        subtext = "Authors with diverse Categories"
        df_arg = author_categories.sort_values(ascending=False)
        self.display_dataframe(subtext, df_arg, 15)

        # Inspect query from the visualization chart
        x = author_categories.values[-15:]
        y = author_categories.index[-15:]
        x_label = "Authors"
        y_label = "Categories"
        self.plot_plotly_bar(x, y, subtext, x_label, y_label)

        # Display all analyzed Dataframes
        self.display_dataframe("Inspection Rating Dataframe", df_temp_rating, 25)
        self.display_dataframe("Inspection Information Dataframe", df_temp_info, 25)

    @staticmethod
    def plot_plotly_bar(x, y, subtext, x_label, y_label):
        # Using Plotly, generate a bar chart
        text = "Bar Plot for " + str(subtext)

        try:
            fig = px.bar(x=x, y=y, orientation='h', color=y)

            fig.update_layout(title={'text': text, 'font': {'size': 23, 'color': '#1c0308'}, 'x': 0.5},
                              xaxis_title={'text': x_label, 'font': {'size': 18, 'color': '#1c0308'}},
                              yaxis_title={'text': y_label, 'font': {'size': 18, 'color': '#1c0308'}})
            fig.show()

        except Exception as exc:
            print("! ", exc)

    def plot_pie(self, df, subtext, labels, explode):
        # Plot pie chart for a specific feature
        text = "Pie Distribution of " + str(subtext)
        self.graph_settings()

        try:
            plt.pie(df, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(text, fontsize=20)
            plt.axis('off')
            plt.legend(loc="upper left")
            plt.show()

    def plot_scatter(self, x, y, subtext, x_label, y_label):
        # Plot pie chart for a specific feature
        text = "Scatter Plot for  " + str(subtext)
        self.graph_settings()

        try:
            plt.scatter(x, y, c='maroon')

        except Exception as exc:
            print("! ", exc)

        finally:
            plt.title(text, fontsize=20)
            plt.grid()
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()

    def plot_dataframe(self, df, subtext, kind, x_label, y_label, graph_type):
        # Create Plots of a Series or Data frame
        text = str(graph_type) + " Plot for " + str(subtext)
        self.graph_settings()

        try:
            df.plot(kind=kind, figsize=(13, 15), color=['#400313', '#2a1842', '#012136', '#062b2b'])

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(text, fontsize=20)
            plt.grid()
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()

    @staticmethod
    def graph_settings():
        # Customizable Set-ups
        plt.figure(figsize=(13, 15))
        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')
        ax = plt.axes()
        ax.set_facecolor("#e6eef1")


if __name__ == "__main__":
    main = FeatureAnalysis()
    main.data_engineering()
    main.data_inspection()
