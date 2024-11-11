from utils import *
from embedding import Embedding
from clustering import Clustering
from measures import Measures
from postprocessing import post_processing

import argparse
import time
import os
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argvs')
    parser.add_argument('-e', type=str, required=True, help='embedding method')
    parser.add_argument('-c', type=str, required=True, help='clustering method')
    parser.add_argument('-k', type=int, required=False, help='number of clusters')
    parser.add_argument('-minpts', type=int, required=False, help='DBSCAN params')
    parser.add_argument('-eps', type=float, required=False, help='DBSCAN params')
    args = parser.parse_args()

    ''' read data '''
    # df = Input.read_csv_as_df(path='../data/ggg_sg.csv', n=200)   # full data, read only first 200 samples
    # df = Input.read_csv_as_df(path='../data/labeled_news_with_entities_60_utf8.csv')
    df = Input.read_csv_as_df(path='../data/ggg_sg_top200_cluster-only.csv')
    # df = df[:-10]
    print(df)
    pd.set_option('display.max_columns', None)      # settings to print all columns

    ''' select relevant features '''
    # df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode']]
    # df = df[['ContextualText', 'Title', 'DateTime', 'Location', 'CountryCode', 'Entities']]

    d = 0
    dn = 0
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
        dn = X_embedding_bert.shape[1]      # dimension of bert dense embedding
        X_embedding_bert = Embedding.binary_sparse_embedding(X_embedding_bert, num_bits=20)
        
        # Part 2: Entities-based embedding
        # raw_entity_set = Input.read_entity_set_csv(path='../data/entities_60.txt')
        # print(f'len(raw_entity_set) = {len(raw_entity_set)}')
        entity_set = Input.read_entity_set_json(path='../data/entities_cluster-only.json')
        print(f'len(entity_set) = {len(entity_set)}')
        entity_mapping = Input.read_entity_mapping_json(path='../data/mapping_cluster-only.json')
        print(f'len(entity_mapping) = {len(entity_mapping)}')
        X_embedding_entity, X_mapped = Embedding.entities_based_embedding(df['entity'], entity_set, entity_mapping)

        print(f'shape X_embedding_entity: {X_embedding_entity.shape}')

        # Combine the two embeddings
        X_embedding = np.concatenate((X_embedding_bert, X_embedding_entity), axis=1)
        print(f'shape X_embedding: {X_embedding.shape}')

    embedding_time = time.time() - start_time

    ''' clustering '''
    start_time = time.time()
    
    if args.k is not None:
        k = args.k
    else:
        k = 5       # set to default value
    
    if args.minpts is not None:
        minpts = args.minpts
    else:
        minpts = 1  # set to default value

    if args.eps is not None:
        eps = args.eps
    else:
        # 55, 1 for tfidf; 31, 1 for BERT; 0.2, 1 for ours
        eps = 55

    if args.c == 'kmeans':
        # kmeans clustering on embeddings
        labels = Clustering.kmeans_clustering(X=X_embedding, n_clusters=k)
    elif args.c == 'kmodes':
        # kprototypes clustering on embeddings
        labels = Clustering.kmodes_clustering(X=X_embedding, n_clusters=k)
    elif args.c == 'kproto':
        labels = Clustering.kprototypes_clustering(X=X_embedding, dn=dn, n_clusters=k)
    elif args.c == 'dbscan':
        # DBSCAN clustering on embeddings
        labels = Clustering.DBSCAN_clustering(X=X_embedding, eps=eps, minpts=minpts)
    elif args.c == 'kfreqitems':
        # kfreqitems clustering on embeddings
        n = X_embedding.shape[0]
        d = X_embedding.shape[1]
        GA = 0.2
        LA = 0.1
        GB = 0.15
        LB = 0.1

        m1 = 8
        h1 = 6
        t  = 200
        m2 = 6
        h2 = 4
        b  = 1
        D   = 100


        # save sparse embeddings to file
        np.savetxt(X=X_embedding, delimiter=' ', fname='../data/X_embedding.txt', fmt='%d')
        P = '../data/X_embedding'
        O = '../results/kfreqitems'

        os.chdir(f'../kfreqitems')
        subprocess.run('module load mpi', shell=True)
        subprocess.run('make clean', shell=True)
        subprocess.run('make -j', shell=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '5'

        shell_command = f"""
            mpirun -n 1 ./silk \
            -alg 1 -s 1 -m 10 \
            -n {n} -d {d} -k {k} \
            -m1 {m1} -h1 {h1} -t {t} -m2 {m2} -h2 {h2} -b {b} -D {D} \
            -GA {GA} -GB {LA} -LA {GB} -LB {LB} \
            -F int32 \
            -P {P} \
            -O {O}
        """

        result = subprocess.run(shell_command, shell=True, env=env)
        if result.returncode != 0:
            print(f'kFreqItems: command failed with return code {result.returncode}')

        labels = Input.read_labels(path=f'{O}/labels_{k}.txt')
        labels_post_proc = post_processing(X_embedding_entity, labels, similarity_threshold=0.05)
        print(f'labels_post_proc = {labels_post_proc}')
        # labels = labels_post_proc
        

    clustering_time = time.time() - start_time

    ''' evaluations '''
    # measures with ground truth label
    # ground truth labels required here
    ground_truth_labels = df['cluster_id'].tolist()
    ground_truth_labels = np.array(ground_truth_labels)
    
    # ground_truth_clusters = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 33], [41, 42]]
    # for i in range(n):
    #     ground_truth_labels[i] = -1
    # cid = 0
    # for cluster in ground_truth_clusters:
    #     for pid in cluster:
    #         ground_truth_labels[pid] = cid
    #     cid += 1

    labels = np.array(labels)
    ari = Measures.safe_ari(ground_truth_labels, labels, filtering=True)
    nmi = Measures.safe_ari(ground_truth_labels, labels, filtering=True)
    purity = Measures.purity(ground_truth_labels, labels, filtering=True)

    # measures without ground truth label


    format_str = "embedding: {:<5s}, clustering: {:<10s} | k: {:<3d}, eps: {:<3.2f}, minpts: {:<3d} \
        | ari: {:<3.2f}, nmi: {:<3.2f}, purity: {:<3.2f} | embedding time: {:<5.2f}, clustering time: {:<5.2f}"
    print(format_str.format(args.e, args.c, k, eps, minpts, ari, nmi, purity, embedding_time, clustering_time) + '\n\n')
    
    ''' interpret the result '''
    print(f'ground truth labels:')
    print(f'{ground_truth_labels.tolist()}\n')

    print(f'predicted labels:')
    print(f'{labels}\n')

    print(f'ground truth clusters:')
    gt_clusters = Utils.get_clusters(ground_truth_labels, nontrivial=False)
    print(f'{gt_clusters}\n')

    print(f'clusters detected:')
    clusters = Utils.get_clusters(labels, nontrivial=False)
    print(f'{clusters}\n')

    if args.e == 'ours':
        for cluster in clusters:
            for id in cluster:
                print(f"{id}: {df['entity'].iloc[id]}")
            print('\n')

        for cluster in clusters:
            for id in cluster:
                print(f"{id} {X_mapped[id]}")
            print('\n')
    
    print(X_mapped)

    

