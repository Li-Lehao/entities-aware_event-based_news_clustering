import numpy as np
import pandas as pd

class Input:
    @staticmethod
    def read_csv_as_df(path, n=None):
        if n is not None:
            df = pd.read_csv(path, nrows=n)     # read only top n rows
        else:
            df = pd.read_csv(path)
        return df
    

class Utils:
    @staticmethod
    def cosine_similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)

    