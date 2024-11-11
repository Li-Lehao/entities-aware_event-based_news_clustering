from utils import *

import math
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

class Embedding:
    @staticmethod
    def tfidf_embedding(X):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_tfidf = vectorizer.fit_transform(X)
        return X_tfidf

    @staticmethod
    def bert_embedding(X):
        ''' get BRET embeddings for a given text '''
        def get_bert_embedding(text, tokenizer, model):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # use the mean of the last hidden state as the embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        bert_embeddings = []
        for text in X:
            embedding = get_bert_embedding(text, tokenizer, model)
            bert_embeddings.append(embedding)

        X_bert = torch.cat(bert_embeddings, dim=0).numpy()
        return X_bert

    @staticmethod
    def entities_based_embedding(X, entity_set, entity_mapping):
        # print(f'entitiy_mapping = {entity_mapping}')
        # map synonyms to the representative one
        X_mapped = []
        for sample in X:
            sample_mapped = []
            # print(f'sample = {sample}')
            sample = sample.strip().split(',')
            for entity in sample:
                entity = entity.strip()
                # print(f'entity = {entity}')
                if entity in entity_mapping:
                    sample_mapped.append(entity_mapping[entity])
            X_mapped.append(sample_mapped)
        # X_mapped = X

        # one-hot encode each sample based on the overall entity set
        m = len(entity_set)
        entity_set = sorted(entity_set)
        entity_to_index = {entity: idx for idx, entity in enumerate(entity_set)}

        X_onehot = []
        for sample in X_mapped:
            sample_onehot = np.zeros(m, dtype=int)
            for entity in sample:
                if entity in entity_to_index:
                    sample_onehot[entity_to_index[entity]] = 1
            X_onehot.append(sample_onehot)
        return np.array(X_onehot), X_mapped
    
    @staticmethod
    def binary_sparse_embedding_single_col(X, num_bits):
        """
        convert dense embedding into binary sparse format, works for a single column
        """
        m = num_bits // 2
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).flatten()     # scale and flatten to 1d array
        
        X_embedded = []
        for sample in X_scaled:
            low = math.floor(sample * m)
            high = low + m
            sample_embedded = np.zeros(num_bits, dtype=int)
            sample_embedded[low:high] = 1       # assign 1s to the range [low, high)
            X_embedded.append(sample_embedded)
        return np.array(X_embedded)
    
    @staticmethod
    def binary_sparse_embedding(X, num_bits):
        """
        Given X with multiple columns, binary sparse embed each column
        """
        X_embedded = []
        for i in range(X.shape[1]):
            column_embedded = Embedding.binary_sparse_embedding_single_col(X[:, i], num_bits=num_bits)
            if i == 0:
                X_embedded = column_embedded
            else:
                X_embedded = np.concatenate((X_embedded, column_embedded), axis=1)
        return X_embedded
    
