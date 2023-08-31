from surprise import Dataset, Reader
from surprise import KNNWithMeans
import joblib as jb
from surprise import NMF, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import os
import sys
sys.path.append(os.path.abspath('../visualization'))
import visualize


class MlUtils:
    def __init__(self):
        self.visualize = visualize.Visualize()
        pass

    @staticmethod
    def partition_data(dataframe, columns, test_size):
        print('-' * 150)

        try:
            if dataframe is not None:
                # Load data from a file
                reader = Reader(line_format='user item rating', sep=',')
                data = Dataset.load_from_df(dataframe[columns], reader=reader)

                # Split the data into training and testing sets
                train_set, test_set = train_test_split(data, test_size=test_size)

                raw_ratings = data.raw_ratings

                print("\n• Raw Values of the modified Dataframe:")
                print(raw_ratings[:5])

                return train_set, test_set

        except Exception as exc:
            print("! Invalid Dataframe. !", exc)
            raise FileNotFoundError("! Dataframe not found. !")

        else:
            print("> Dataframe loaded successfully.")

        finally:
            print('-' * 150)

    @staticmethod
    def construct_model(algorithm, train_data, label_type):
        model = ''

        if algorithm == 'KNNWithMeans':
            # Train an KNN With Means model on the specified score
            model = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
            model.fit(train_data)

        elif algorithm == 'NMF':
            # Train an NMF With Means model on the specified score
            model = NMF(n_factors=20, random_state=42)
            model.fit(train_data)

        elif algorithm == 'SVD':
            # Train an SVD With Means model on the specified score
            model = SVD(n_factors=70, random_state=42)
            model.fit(train_data)

        try:
            jb.dump(model, f'../../models/{algorithm}_{label_type}.pkl')

        except Exception as exc:
            print("! Exception encountered", exc)

        else:
            print("• Model saved successfully", '')

    def model_test_case(self, algorithm, train_data, test_data):
        model = ''
        r_mse_list = []
        mse_list = []
        mae_list = []

        if algorithm == 'KNNWithMeans':
            title = 'Knn Hyperparameter Evaluation'
            x_label = 'Number of k-points'

            for i in range(10, 80, 10):
                # Train an KNN With Means model on the specified score
                model = KNNWithMeans(k=i, sim_options={'name': 'pearson_baseline', 'user_based': True})
                model.fit(train_data)

                r_mse, mse, mae = self.evaluate_model(model, test_data, return_value=True)

                r_mse_list.append(r_mse)
                mse_list.append(mse)
                mae_list.append(mae)

        elif algorithm == 'SVD':
            title = 'SVD Hyperparameter Evaluation'
            x_label = 'Number of Factors'

            # Train an SVD With Means model on the specified score
            for i in range(10, 80, 10):
                model = SVD(n_factors=i, random_state=42)
                model.fit(train_data)

                r_mse, mse, mae = self.evaluate_model(model, test_data, return_value=True)

                r_mse_list.append(r_mse)
                mse_list.append(mse)
                mae_list.append(mae)

        elif algorithm == 'NMF':
            title = 'NMF Hyperparameter Evaluation'
            x_label = 'Number of Factors'

            # Train an SVD With Means model on the specified score
            for i in range(10, 80, 10):
                model = NMF(n_factors=i, random_state=42)
                model.fit(train_data)

                r_mse, mse, mae = self.evaluate_model(model, test_data, return_value=True)

                r_mse_list.append(r_mse)
                mse_list.append(mse)
                mae_list.append(mae)

        x = range(10, 80, 10)
        self.visualize.plot_graph(x, r_mse_list, title, x_label, 'RMSE')
        self.visualize.plot_graph(x, mse_list, title, x_label, 'MSE')
        self.visualize.plot_graph(x, mae_list, title, x_label, 'MAE')

    @staticmethod
    def load_model(model_path):
        if model_path:
            model = jb.load(model_path)
            return model

        else:
            raise FileNotFoundError("! Model Path not found. !")

    @staticmethod
    def evaluate_model(model, test_data, return_value=False):

        predictions = model.test(test_data)
        r_mse = accuracy.rmse(predictions)
        mse = accuracy.mse(predictions)
        mae = accuracy.mae(predictions)

        print("\n• Root Mean Square Error:", r_mse)
        print("\n• Standard Mean Square Error:", mse)
        print("\n• Standard Mean Absolute Error:", mae)
        print(predictions[:5])

        if return_value:
            return r_mse, mse, mae


if __name__ == "__main__":
    main = MlUtils()
