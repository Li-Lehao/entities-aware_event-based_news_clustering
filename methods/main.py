from utils import *
from embedding import Embedding
from clustering import Clustering

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argvs')
    parser.add_argument('-e', type=str, required=True, help='embedding method')
    parser.add_argument('-c', type=str, required=True, help='clustering method')
    args = parser.parse_args()

    ''' read data '''
    df = Input.read_csv_as_df(path='../data/ggg_sg.csv', n=2)
    pd.set_option('display.max_columns', None)      # settings to print all columns

    ''' select relevant features '''
    df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode']]

    ''' embedding '''
    if args.e == 'tfidf':
        df['TextForEmbedding'] = df['Title'] + ' ' + df['ContextualText']
        X_embedding = Embedding.tfidf_embedding(df['TextForEmbedding'])

    ''' clustering '''
    if args.c == 'kmeans':
        labels = Clustering.kmeans_clustering(X=X_embedding, n_clusters=10)
    elif args.c == 'dbscan':
        labels = Clustering.DBSCAN_clustering(X=X_embedding, eps=0.5, minpts=5)

    ''' evaluations '''
    # measures with ground truth label
    # ground truth labels required here

    # measures without ground truth label
    

    





