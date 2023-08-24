import pandas as pd
import missingno as msn
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.font_manager import FontProperties
import plotly.express as px
from scipy import stats
import numpy as np
import plotly.graph_objects as go
import random as rnd


class Visualize:
    def __init__(self):
        pass

    @staticmethod
    def plot_multi_histogram(d1, d2, bins, title_d1, title_d2):
        fig, axes = plt.subplots(1, 2)
        plt.rcParams["figure.figsize"] = [14, 16]
        plt.rcParams["figure.autolayout"] = True

        try:
            d1.hist(bins=bins, color='#800e1b', ax=axes[0], edgecolor='#0d0103', linewidth=1.2)
            d2.hist(bins=bins, color='#06701e', ax=axes[1], edgecolor='#0d0103', linewidth=1.2)

        except Exception as exc:
            print("! ", exc)

        else:
            axes[0].set_title(title_d1, fontsize=12, fontweight='bold')
            axes[1].set_title(title_d2, fontsize=12, fontweight='bold')
            plt.show()

    @staticmethod
    def plot_kde(df, text, x_label, y_label):
        try:
            df.plot(kind='kde', figsize=(14, 16), color='#02659e')

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(text, fontsize=16, fontweight='bold')
            plt.xlabel(x_label, fontsize=12, fontweight='bold')
            plt.ylabel(y_label, fontsize=12, fontweight='bold')
            plt.grid()
            plt.show()

    @staticmethod
    def plot_correlation(df):
        title = "Heatmap: Selective Continuous Features"

        try:
            data_plot = sb.heatmap(df.corr(), cmap="RdGy", annot=True)

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(title, fontsize=16, fontweight='bold')
            plt.yticks(rotation='horizontal')
            plt.show()

    def plot_pie(self, df, explode, labels, title, bbox_to_anchor):
        self.graph_settings()

        try:
            plt.pie(df, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(title, fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.legend(bbox_to_anchor=bbox_to_anchor, fontsize="15", loc="upper right")
            plt.show()

    def plot_dataframe(self, df, text, kind, x_label, y_label):
        # Create Plots of a Series or a Data frame object
        self.graph_settings()

        try:
            df.plot(kind=kind, figsize=(13, 15), color=['#400313', '#2a1842', '#012136', '#062b2b'])

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(text, fontsize=20, fontweight='bold')
            plt.grid()
            plt.xlabel(x_label, fontsize=12, fontweight='bold')
            plt.ylabel(y_label, fontsize=12, fontweight='bold')
            plt.show()

    def plot_scatter(self, x, y, text, x_label, y_label):
        # Plot Scatter plot for a specific set of features
        self.graph_settings()

        try:
            plt.scatter(x, y, c='#80450e', edgecolor='#0d0103', linewidth=0.5)

        except Exception as exc:
            print("! ", exc)

        finally:
            plt.title(text, fontsize=16, fontweight='bold')
            plt.xlabel(x_label, fontsize=12, fontweight='bold')
            plt.ylabel(y_label, fontsize=12, fontweight='bold')
            plt.grid()
            plt.show()

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

    def plot_bar(self, df, title, x_label, y_label, orientation):
        self.graph_settings()
        color_palette = ['#047678', '#7a3100', '#82102a']
        rand = rnd.randint(0, 2)

        # Construct a Bar plot for clarity
        try:
            if orientation == 'h':
                plt.barh(df, c=color_palette[rand], edgecolor='#0d0103', linewidth=0.5)

            else:
                plt.bar(df, c='#7a3100', edgecolor='#0d0103', linewidth=0.5)

        except Exception as exc:
            print("! ", exc)

        else:
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel(x_label, fontsize=12, fontweight='bold')
            plt.ylabel(y_label, fontsize=12, fontweight='bold')
            plt.grid()
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
    main = Visualize()
