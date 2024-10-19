import numpy as np
import pandas as pd
import json

class Input:
    @staticmethod
    def read_csv_as_df(path, n=None):
        if n is not None:
            df = pd.read_csv(path, nrows=n)     # read only top n rows
        else:
            df = pd.read_csv(path)
        return df
    
    @staticmethod
    def read_entity_set_csv(path):
        with open(path, 'r') as file:
            content = file.read()
        entities = [entity.strip() for entity in content.split(',')]
        return np.array(entities)
    
    @staticmethod
    def read_entity_set_json(path):
        with open(path, 'r') as file:
            data = json.load(file)
        entity_set = data['entities']
        return entity_set
    
    @staticmethod
    def read_entity_mapping_json(path):
        with open(path, 'r') as file:
            data = json.load(file)
        entity_mapping = data['mapping']
        return entity_mapping

class Utils:
    @staticmethod
    def cosine_similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)

    