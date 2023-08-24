import os
import sys
import pandas as pd
import joblib as jb
sys.path.append(os.path.abspath('../'))
from py_utils import generic_utils
from py_utils import ml_utils


class KnnTest:
    def __init__(self):
        self.current_model = 'SVD'
        self.data_type = 'base'
        self.gu = generic_utils.GenericUtils()
        self.ml = ml_utils.MlUtils()
        self.df_book_ratings = pd.read_csv('../../data/test_data.csv')

    def test_model(self):
        # Load Model Parameters
        df_book_ratings = self.df_book_ratings[['user_id', 'book_id', 'rating']]
        test_data = df_book_ratings.to_numpy()
        model = self.ml.load_model(f'../../models/{self.current_model}_{self.data_type}.pkl')
        self.ml.evaluate_model(model, test_data)


if __name__ == "__main__":
    main = KnnTest()
    main.test_model()
