from collections import defaultdict
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
    
    @staticmethod
    def read_labels(path):
        with open(path, 'r') as file:
            labels = [int(line.strip()) for line in file]
        return labels

class Utils:
    @staticmethod
    def cosine_similarity(A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return dot_product / (norm_A * norm_B)
    
    @staticmethod
    def get_clusters(labels, nontrivial=False):
        clusters = defaultdict(list)
        
        # Group indices by label
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            clusters[label].append(idx)
        
        # If nontrivial is True, filter out clusters of size 1
        if nontrivial:
            clusters = {label: indices for label, indices in clusters.items() if len(indices) > 1}
        
        return list(clusters.values())

    