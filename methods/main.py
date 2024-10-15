from utils import *
from embedding import Embedding
from clustering import Clustering

import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argvs')
    parser.add_argument('-e', type=str, required=True, help='embedding method')
    parser.add_argument('-c', type=str, required=True, help='clustering method')
    args = parser.parse_args()

    ''' read data '''
    df = Input.read_csv_as_df(path='../data/ggg_sg.csv', n=200)
    pd.set_option('display.max_columns', None)      # settings to print all columns

    ''' select relevant features '''
    # df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode']]
    # df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode', 'Entities']]

    ''' embedding '''
    start_time = time.time()
    if args.e == 'tfidf':
        df['TextForEmbedding'] = df['Title'] + ' ' + df['ContextualText']
        X_embedding = Embedding.tfidf_embedding(df['TextForEmbedding'])
    elif args.e == 'bert':
        df['TextForEmbedding'] = df['Title'] + ' ' + df['ContextualText']
        X_embedding = Embedding.bert_embedding(df['TextForEmbedding'])
    embedding_time = time.time() - start_time

    ''' clustering '''
    start_time = time.time()
    if args.c == 'kmeans':
        labels = Clustering.kmeans_clustering(X=X_embedding, n_clusters=10)
    elif args.c == 'dbscan':
        labels = Clustering.DBSCAN_clustering(X=X_embedding, eps=70, minpts=3)
    clustering_time = time.time() - start_time

    ''' evaluations '''
    # measures with ground truth label
    # ground truth labels required here

    # measures without ground truth label
    print(f'labels = {labels}')
