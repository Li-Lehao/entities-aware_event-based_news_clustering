from utils import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class Embedding:
    @staticmethod
    def tfidf_embedding(X):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_tfidf = vectorizer.fit_transform(X)
        return X_tfidf

