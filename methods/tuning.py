from utils import *
from embedding import Embedding
from clustering import Clustering
from measures import Measures

import argparse
import time
import os
import subprocess


def tuning_ours():
    ''' read data '''
    df = Input.read_csv_as_df(path='../data/ggg_sg_top200_cluster-only.csv')

    ''' embedding '''
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

    ''' clustering '''
    # save sparse embeddings to file
    np.savetxt(X=X_embedding, delimiter=' ', fname='../data/X_embedding.txt', fmt='%d')
    
    # env
    os.chdir(f'../kfreqitems')
    subprocess.run('module load mpi', shell=True)
    subprocess.run('make clean', shell=True)
    subprocess.run('make -j', shell=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '5'

    # params
    P = '../data/X_embedding'
    O = '../results/kfreqitems'
    n = X_embedding.shape[0]
    d = X_embedding.shape[1]
    GA_list = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]
    GB_list = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]
    m1_list = [4, 6, 8]
    h1_list = [4, 6, 8]
    t = 200
    m2_list = [4, 6, 8]
    h2_list = [4, 6, 8]
    k_list = [8, 10, 15]
    b = 1
    D = 100

    # grid search
    best_ari = 0
    best_params = []

    for k in k_list:
        for GA in GA_list:
            for GB in GB_list:
                for m1 in m1_list:
                    for h1 in h1_list:
                        for m2 in m2_list:
                            for h2 in h2_list:
                                # run kfreqitems
                                shell_command = f"""
                                    mpirun -n 1 ./silk \
                                    -alg 1 -s 1 -m 10 \
                                    -n {n} -d {d} -k {k} \
                                    -m1 {m1} -h1 {h1} -t {t} -m2 {m2} -h2 {h2} -b {b} -D {D} \
                                    -GA {GA} -GB {GA} -LA {GB} -LB {GB} \
                                    -F int32 \
                                    -P {P} \
                                    -O {O}
                                """

                                result = subprocess.run(shell_command, shell=True, env=env)
                                if result.returncode != 0:
                                    print(f'kFreqItems: command failed with return code {result.returncode}')

                                labels = Input.read_labels(path=f'{O}/labels_{k}.txt')

                                # evaluation
                                ground_truth_labels = df['cluster_id'].tolist()
                                ground_truth_labels = np.array(ground_truth_labels)
                                labels = np.array(labels)
                                ari = Measures.safe_ari(ground_truth_labels, labels, filtering=True)
                                nmi = Measures.safe_ari(ground_truth_labels, labels, filtering=True)
                                purity = Measures.purity(ground_truth_labels, labels, filtering=True)

                                params = [k, GA, GB, m1, h1, m2, h2]
                                if ari > best_ari:
                                    best_ari = ari
                                    best_params = params
                                with open('../results/tuning/ours.txt', 'a+') as file:
                                    file.write(f'{params}, {ari}, {nmi}, {purity}\n')
    print(f'best_ari = {best_ari}, best_params = {best_params}')




if __name__ == '__main__':
    tuning_ours()
