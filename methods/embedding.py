from utils import *

from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
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
