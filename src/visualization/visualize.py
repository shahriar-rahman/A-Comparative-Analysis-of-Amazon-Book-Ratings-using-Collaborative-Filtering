import pandas as pd
import missingno as msn
from matplotlib import pyplot as plt
import seaborn as sb
from matplotlib.font_manager import FontProperties
import plotly.express as px
import plotly.graph_objects as go


class Visualize:
    def __init__(self):
        pass

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
    main = Visualize()
