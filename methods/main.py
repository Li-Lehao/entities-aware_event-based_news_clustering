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
    # df = Input.read_csv_as_df(path='../data/ggg_sg.csv', n=200)   # full data, read only first 200 samples
    df = Input.read_csv_as_df(path='../data/labeled_news_with_entities_60_utf8.csv')
    pd.set_option('display.max_columns', None)      # settings to print all columns

    ''' select relevant features '''
    # df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode']]
    # df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode', 'Entities']]

    ''' embedding '''
    start_time = time.time()
    if args.e == 'tfidf':
        # TF-IDF embedding on title + contextual text
        df['TextForEmbedding'] = df['Title'] + ' ' + df['ContextualText']
        X_embedding = Embedding.tfidf_embedding(df['TextForEmbedding'])
    elif args.e == 'bert':
        # BERT embedding on title + contextual text
        df['TextForEmbedding'] = df['Title'] + ' ' + df['ContextualText']
        X_embedding = Embedding.bert_embedding(df['TextForEmbedding'])
    elif args.e == 'ours':
        # Part 1: BERT embedding on title + contextual text; Conversion to binary format required
        df['TextForEmbedding'] = df['Title'] + ' ' + df['ContextualText']
        X_embedding_bert = Embedding.bert_embedding(df['TextForEmbedding'])
        X_embedding_bert_binary = Embedding.binary_sparse_embedding(X_embedding_bert, num_bits=20)
        
        # Part 2: Entities-based embedding
        raw_entity_set = Input.read_entity_set_csv(path='../data/entities_60.txt')
        print(f'len(raw_entity_set) = {len(raw_entity_set)}')
        entity_set = Input.read_entity_set_json(path='../data/entities_60.json')
        print(f'len(entity_set) = {len(entity_set)}')
        entity_mapping = Input.read_entity_mapping_json(path='../data/mapping_60.json')
        print(f'len(entity_mapping) = {len(entity_mapping)}')
        X_embedding_entity = Embedding.entities_based_embedding(df['entity'], entity_set, entity_mapping)
        print(f'shape entity based embedding: {X_embedding_entity.shape}')

        # Combine the two embeddings
        X_embedding = np.concatenate((X_embedding_bert, X_embedding_entity), axis=1)

    embedding_time = time.time() - start_time

    ''' clustering '''
    start_time = time.time()
    if args.c == 'kmeans':
        # kmeans clustering on embeddings
        labels = Clustering.kmeans_clustering(X=X_embedding, n_clusters=10)
    elif args.c == 'dbscan':
        # DBSCAN clustering on embeddings
        labels = Clustering.DBSCAN_clustering(X=X_embedding, eps=70, minpts=3)
    clustering_time = time.time() - start_time

    ''' evaluations '''
    # measures with ground truth label
    # ground truth labels required here
    

    # measures without ground truth label
    print(f'labels = {labels}')
