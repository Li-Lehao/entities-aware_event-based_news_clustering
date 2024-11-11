#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>

#include "def.h"
#include "util.cuh"
#include "lsh.cuh"
#include "bucket.h"
#include "bin.h"
#include "assign.cuh"
#include "eval.h"

namespace clustering {

// -----------------------------------------------------------------------------
void generate_k_distinct_ids(       // generate k distinct ids
    int k,                              // k value
    int n,                              // total range
    int *distinct_ids);                 // distinct ids (return)

// -----------------------------------------------------------------------------
template<class DType>
void distinct_ids_to_local_seed(    // get local seeds from distinct ids
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of data points
    int   k,                            // top-k value
    const int   *distinct_ids,          // distinct ids
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    std::vector<int> &local_seedset,    // local seed set (return)
    std::vector<f32> &local_seedwgt,    // local seed weight (return)
    std::vector<u64> &local_seedpos,    // local seed position (return)
    std::vector<f32> &local_seedsum)    // local seed sum of weights (return)
{
    local_seedset.reserve(k*avg_d); // estimate, not correct
    local_seedwgt.reserve(k*avg_d); // estimate, not correct
    local_seedpos.reserve(k+1);     // estimate, not correct
    local_seedsum.reserve(k);       // estimate, not correct
    
    local_seedpos.push_back(0);
    
    int lower_bound = n * rank;
    int upper_bound = n + lower_bound;
    int id = -1; 
    for (int i = 0; i < k; ++i) {
        id = distinct_ids[i];
        if (id < lower_bound || id >= upper_bound) continue;
        
        // get the local data (by id)
        id -= lower_bound;
        const DType *coord = dataset + datapos[id];
        const float *weigh = datawgt + datapos[id];
        int   len = get_length(id, datapos);
        float sum = datasum[id];
        
        // add this data to local seedset and seedpos
        local_seedset.insert(local_seedset.end(), coord, coord+len);
        local_seedwgt.insert(local_seedwgt.end(), weigh, weigh+len);
        local_seedpos.push_back(len);
        local_seedsum.push_back(sum);
    }
}

// -----------------------------------------------------------------------------
void gather_all_local_seedset(      // gather all local seedset to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    const std::vector<int> &local_seedset, // local seedset
    std::vector<int> &seedset);         // seedset at root (return)

// -----------------------------------------------------------------------------
void gather_all_local_seedwgt(      // gather all local seedwgt to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    const std::vector<f32> &local_seedwgt, // local seedwgt
    std::vector<f32> &seedwgt);         // seedwgt at root (return)

// -----------------------------------------------------------------------------
void gather_all_local_seedpos(      // gather all local seedpos to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   k,                            // k value
    const std::vector<u64> &local_seedpos, // local seedpos
    std::vector<u64> &seedpos);         // seedpos at root (return)

// -----------------------------------------------------------------------------
void gather_all_local_seedsum(      // gather all local seedsum to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   k,                            // k value
    const std::vector<f32> &local_seedsum, // local seedsum
    std::vector<f32> &seedsum);         // seedsum at root (return)

// -----------------------------------------------------------------------------
template<class DType>
void get_k_seeds(                   // get k seeds based on distinct ids
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    const int   *distinct_ids,          // k distinct ids
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // clear space
    std::vector<int>().swap(seedset);
    std::vector<f32>().swap(seedwgt);
    std::vector<u64>().swap(seedpos);
    std::vector<f32>().swap(seedsum);
    
    // -------------------------------------------------------------------------
    //  get local seedset and seedpos
    // -------------------------------------------------------------------------
    std::vector<int> local_seedset;
    std::vector<f32> local_seedwgt;
    std::vector<u64> local_seedpos;
    std::vector<f32> local_seedsum;
    
    distinct_ids_to_local_seed<DType>(rank, size, n, avg_d, k, distinct_ids,
        dataset, datawgt, datapos, datasum, local_seedset, local_seedwgt, 
        local_seedpos, local_seedsum);
    
    // -------------------------------------------------------------------------
    //  get global seedset and seedpos to root
    // -------------------------------------------------------------------------
    if (size == 1) {
        // single-thread case: directly swap local seed and seed
        seedset.swap(local_seedset);
        seedwgt.swap(local_seedwgt);
        seedpos.swap(local_seedpos);
        seedsum.swap(local_seedsum);
    }
    else {
        // gather all local seeds into global seeds to root
        gather_all_local_seedset(rank, size, local_seedset, seedset);
        gather_all_local_seedwgt(rank, size, local_seedwgt, seedwgt);
        gather_all_local_seedpos(rank, size, k, local_seedpos, seedpos);
        gather_all_local_seedsum(rank, size, k, local_seedsum, seedsum);
    
        // broadcast global seeds from root to other threads
        broadcast_seeds(rank, size, seedset, seedwgt, seedpos, seedsum);
        
        std::vector<int>().swap(local_seedset);
        std::vector<f32>().swap(local_seedwgt);
        std::vector<u64>().swap(local_seedpos);
        std::vector<f32>().swap(local_seedsum);
    }
    // accumulate the length of seeds to get the start position of each seed
    for (int i = 1; i <= k; ++i) seedpos[i] += seedpos[i-1];
}

// -----------------------------------------------------------------------------
template<class DType>
void random_seeding(                // init k centers by random seeding
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    srand(RANDOM_SEED); // fix a random seed
    
    // -------------------------------------------------------------------------
    //  generate k distinct ids from [0, N-1]
    // -------------------------------------------------------------------------
    if (rank == 0) generate_k_distinct_ids(k, N, distinct_ids);
    if (size > 1) {
        // multi-thread case: broadcast distinct_ids from root to other threads
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(distinct_ids, k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the k distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(rank, size, n, avg_d, k, distinct_ids, dataset, datawgt,
        datapos, datasum, seedset, seedwgt, seedpos, seedsum);
}

// -----------------------------------------------------------------------------
void broadcast_target_data(         // broadcast target data to all threads
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    std::vector<int> &target_coord,     // target coord (return)
    std::vector<f32> &target_weigh,     // target weigh (return)
    float &target_sum);                 // target sum   (return)

// -----------------------------------------------------------------------------
template<class DType>
void get_data_by_id(                // get a data by input id
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   id,                           // input data id
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    std::vector<int> &target_coord,     // target data coord  (return)
    std::vector<f32> &target_weigh,     // target data weight (return)
    float &target_sum)                  // target data sum og weights (return)
{
    // clear space
    std::vector<int>().swap(target_coord);
    std::vector<f32>().swap(target_weigh);
    target_sum = 0.0f;
    
    // get local target data
    int lower_bound = n * rank;
    int upper_bound = n + lower_bound;
    if (lower_bound <= id && id < upper_bound) {
        // retrieve the target data
        id -= lower_bound;
        
        const DType *coord = dataset + datapos[id];
        const float *weigh = datawgt + datapos[id];
        int len = get_length(id, datapos);
        
        target_sum = datasum[id];
        target_coord.reserve(len);
        target_coord.insert(target_coord.end(), coord, coord+len);
        target_weigh.reserve(len);
        target_weigh.insert(target_weigh.end(), weigh, weigh+len);
    }
    // if multi-thread case: broadcast the target data to all threads
    if (size > 1) {
        broadcast_target_data(rank, size, target_coord, target_weigh, target_sum);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_one_nn_dist(            // update a single nn_dist for input data
    int   did,                          // input data id
    int   n_seed,                       // length of input seed
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seed_coord,            // input seed coord
    const float *seed_weigh,            // input seed weigh
    float seed_sum,                     // last seed sum of weights
    float &nn_dist)                     // nn_dist (return)
{
    int   n_data = get_length(did, datapos);
    float data_sum = datasum[did];
    const DType *data_coord = dataset + datapos[did];
    const float *data_weigh = datawgt + datapos[did];
    
    float dist = weighted_jaccard_dist<DType>(n_data, n_seed, data_sum, 
        seed_sum, data_coord, data_weigh, seed_coord, seed_weigh);
    
    if (nn_dist > dist) nn_dist = dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data by a seed
    int   n,                            // number of data points
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seed_coord, // last seed coord
    const std::vector<f32> &seed_weigh, // last seed weigh
    float seed_sum,                     // last seed sum of weights
    float *nn_dist)                     // nn_dist (return)
{
    int n_seed = (int) seed_coord.size();
    const int *seed_coord_ptr = (const int*) seed_coord.data();
    const f32 *seed_weigh_ptr = (const f32*) seed_weigh.data();
    
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        update_one_nn_dist<DType>(i, n_seed, dataset, datawgt, datapos, datasum,
            seed_coord_ptr, seed_weigh_ptr, seed_sum, nn_dist[i]);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seed_coord, // last seed coord
    const std::vector<f32> &seed_weigh, // last seed weigh
    float seed_sum,                     // last seed sum of weights
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template<class DType>
void update_dist_and_prob(          // update nn_dist and prob by last seed
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const int   *weights,               // weights of data set
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seed_coord, // last seed coordinate
    const std::vector<f32> &seed_weigh, // last seed weight
    float seed_sum,                     // last seed sum of weights
    float *nn_dist,                     // nn_dist (return)
    float *prob)                        // probability (return)
{
    // -------------------------------------------------------------------------
    //  update nn_dist for the local data
    // -------------------------------------------------------------------------
    if (hard_device == 0) {
        // use OpenMP to update nn_dist (by default)
        update_nn_dist<DType>(n, dataset, datawgt, datapos, datasum, 
            seed_coord, seed_weigh, seed_sum, nn_dist);
    }
    else {
        // use GPUs to update nn_dist
        update_nn_dist<DType>(rank, n, dataset, datawgt, datapos, datasum, 
            seed_coord, seed_weigh, seed_sum, nn_dist);
    }
    
    // -------------------------------------------------------------------------
    //  get global nn_dist to root
    // -------------------------------------------------------------------------
    float *all_nn_dist = new float[N];
    if (size == 1) {
        // single-thread case: directly copy one to another
        std::copy(nn_dist, nn_dist + n, all_nn_dist);
    }
    else {
        // multi-thread case: gather nn_dist to root
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(nn_dist, n, MPI_FLOAT, all_nn_dist, n, MPI_FLOAT, 0, 
            MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // -------------------------------------------------------------------------
    //  @root: update global probability array
    // -------------------------------------------------------------------------
    if (rank == 0) {
        prob[0] = weights[0] * SQR(all_nn_dist[0]);
        for (int i = 1; i < N; ++i) {
            prob[i] = prob[i-1] + weights[i]*SQR(all_nn_dist[i]);
        }
    }
    delete[] all_nn_dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeanspp_seeding(              // init k centers by k-means++
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *weights,               // weights of data points
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    srand(RANDOM_SEED); // fix a random seed
    
    // -------------------------------------------------------------------------
    //  init nn_dist array & probability array
    // -------------------------------------------------------------------------
    float *nn_dist = new float[n];
    for (int i = 0; i < n; ++i) nn_dist[i] = MAX_FLOAT;
        
    float *prob = new float[N];
    prob[0] = (float) weights[0];
    for (int i = 1; i < N; ++i) prob[i] = prob[i-1] + weights[i];
    
    // -------------------------------------------------------------------------
    //  sample the first center
    // -------------------------------------------------------------------------
    int id = -1;
    // @root: sample the 1st center uniformly at random
    if (rank == 0) {
        float val = uniform(0.0f, prob[N-1]);
        id = std::lower_bound(prob, prob+N, val) - prob;
    }
    // broadcast id from root to other threads if multi-thread case
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    distinct_ids[0] = id;
    
    // -------------------------------------------------------------------------
    //  sample the remaining (k-1) centers by D^2 sampling
    // -------------------------------------------------------------------------
    std::vector<int> last_seed_coord;
    std::vector<f32> last_seed_weigh;
    float last_seed_sum = 0.0f;
    
    for (int i = 1; i < k; ++i) {
        // get the last seed by id
        get_data_by_id<DType>(rank, size, n, id, dataset, datawgt, datapos, 
            datasum, last_seed_coord, last_seed_weigh, last_seed_sum);
        
        // update nn_dist and prob by last seed
        update_dist_and_prob<DType>(rank, size, n, N, hard_device, weights, 
            dataset, datawgt, datapos, datasum, last_seed_coord, 
            last_seed_weigh, last_seed_sum, nn_dist, prob);
        
        // @root: sample the i-th center (id) by D^2 sampling
        if (rank == 0) {
            float val = uniform(0.0f, prob[N-1]);
            id = std::lower_bound(prob, prob+N, val) - prob;
        }
        // broadcast id from root to other threads if multi-thread case
        if (size > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        distinct_ids[i] = id;
        
#ifdef DEBUG_INFO
        if (rank==0 && (i+1)%100==0) printf("Rank #%d: %d/%d\n", rank, i+1, k);
#endif
    }
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the k distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(rank, size, n, avg_d, k, distinct_ids, dataset, datawgt, 
        datapos, datasum, seedset, seedwgt, seedpos, seedsum);

    // release space
    std::vector<int>().swap(last_seed_coord);
    std::vector<f32>().swap(last_seed_weigh);
    delete[] nn_dist;
    delete[] prob;
}

// -----------------------------------------------------------------------------
template<class DType>
void update_knn_dist(               // update knn dist for input data
    int   did,                          // input data id
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seedset
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seedpos
    const float *seedsum,               // seed sum of weights
    float &nn_dist)                     // nn_dist (return)
{
    // retrieve this data by id
    int   n_data   = get_length(did, datapos);
    float data_sum = datasum[did];
    const DType *data_coord = dataset + datapos[did];
    const float *data_weigh = datawgt + datapos[did];
    
    // compare with k seeds and update nn_dist
    for (int i = 0; i < k; ++i) {
        int   n_seed   = get_length(i, seedpos);
        float seed_sum = seedsum[i];
        const int *seed_coord = seedset + seedpos[i];
        const f32 *seed_weigh = seedwgt + seedpos[i];
        
        float dist = weighted_jaccard_dist<DType>(n_data, n_seed, data_sum,
            seed_sum, data_coord, data_weigh, seed_coord, seed_weigh);
        
        if (nn_dist > dist) nn_dist = dist;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data by a seed
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    float *nn_dist)                     // nn_dist (return)
{
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        update_knn_dist<DType>(i, k, dataset, datawgt, datapos, datasum,
            seedset, seedwgt, seedpos, seedsum, nn_dist[i]);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void update_nn_dist(                // update nn_dist for local data by k seeds
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template<class DType>
void update_dist_and_prob(          // update nn_dist & prob by seedset & seedpos
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seedset,    // seed set
    const std::vector<f32> &seedwgt,    // seed weight
    const std::vector<u64> &seedpos,    // seed position
    const std::vector<f32> &seedsum,    // seed sum of weights
    float *nn_dist,                     // nn_dist (return)
    float *prob)                        // probability (return)
{
    // -------------------------------------------------------------------------
    //  update nn_dist for the local data
    // -------------------------------------------------------------------------
    int k = (int) seedpos.size() - 1;
    if (hard_device == 0) {
        // use OpenMP to update nn_dist
        update_nn_dist<DType>(n, k, dataset, datawgt, datapos, datasum, 
            seedset.data(), seedwgt.data(), seedpos.data(), seedsum.data(),
            nn_dist);
    }
    else {
        // use GPUs to update nn_dist (by default)
        update_nn_dist<DType>(rank, n, k, dataset, datawgt, datapos, datasum,
            seedset.data(), seedwgt.data(), seedpos.data(), seedsum.data(),
            nn_dist);
    }
    // -------------------------------------------------------------------------
    //  get global nn_dist to root
    // -------------------------------------------------------------------------
    float *all_nn_dist = new float[N];
    if (size == 1) {
        // single-thread case: directly copy one to another
        assert(n == N);
        std::copy(nn_dist, nn_dist + n, all_nn_dist);
    }
    else {
        // multi-thread case: gather nn_dist to root
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(nn_dist, n, MPI_FLOAT, all_nn_dist, n, MPI_FLOAT, 0, 
            MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // -------------------------------------------------------------------------
    //  @root: update global probability array to all threads
    // -------------------------------------------------------------------------
    if (rank == 0) {
        prob[0] = SQR(all_nn_dist[0]);
        for (int i = 1; i < N; ++i) prob[i] = prob[i-1] + SQR(all_nn_dist[i]);
    }
    delete[] all_nn_dist;
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeansll_overseeding(          // get (tl+1) centers by k-means|| overseeding
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   l,                            // oversampling factor
    int   t,                            // number of iterations
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    int   *all_distinct_ids,            // (t*l+1) distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    srand(RANDOM_SEED); // fix a random seed

    // -------------------------------------------------------------------------
    //  init nn_dist array & probability array
    // -------------------------------------------------------------------------
    float *nn_dist = new float[n];
    for (int i = 0; i < n; ++i) nn_dist[i] = MAX_FLOAT;
        
    float *prob = new float[N];
    for (int i = 0; i < N; ++i) prob[i] = i;
    
    // -------------------------------------------------------------------------
    //  sample the first center
    // -------------------------------------------------------------------------
    int id = -1;
    // @root: sample the 1st center uniformly at random
    if (rank == 0) {
        float val = uniform(0.0f, prob[N-1]);
        id = std::lower_bound(prob, prob+N, val) - prob;
    }
    // broadcast id from root to other threads if multi-thread case
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    all_distinct_ids[0] = id;
    
    // -------------------------------------------------------------------------
    //  sample the remaining (t*l) centers in t iterations by D^2 sampling
    // -------------------------------------------------------------------------
    std::vector<int> last_seedset;
    std::vector<f32> last_seedwgt;
    std::vector<u64> last_seedpos;
    std::vector<f32> last_seedsum;
    
    for (int i = 0; i < t; ++i) {
#ifdef DEBUG_INFO
        printf("Rank #%d: %d/%d\n", rank, i+1, t);
#endif
        // get the last l seeds
        if (i == 0) {
            float tmp_seed_sum = 0.0f;
            get_data_by_id<DType>(rank, size, n, id, dataset, datawgt, datapos, 
                datasum, last_seedset, last_seedwgt, tmp_seed_sum);
            
            std::vector<u64>().swap(last_seedpos);
            last_seedpos.push_back(0);
            last_seedpos.push_back(last_seedset.size());
            
            std::vector<f32>().swap(last_seedsum);
            last_seedsum.push_back(tmp_seed_sum);
        }
        else {
            int *distinct_ids = &all_distinct_ids[(i-1)*l+1];
            
            get_k_seeds<DType>(rank, size, n, avg_d, l, distinct_ids, dataset, 
                datawgt, datapos, datasum, last_seedset, last_seedwgt, 
                last_seedpos, last_seedsum);
        }
        // update nn_dist and prob by the last l seeds
        update_dist_and_prob<DType>(rank, size, n, N, hard_device, dataset, 
            datawgt, datapos, datasum, last_seedset, last_seedwgt, last_seedpos, 
            last_seedsum, nn_dist, prob);
        
        // @root: sample l centers (distinct_ids) by D^2 sampling
        int *distinct_ids = &all_distinct_ids[i*l+1];
        if (rank == 0) {
            for (int j = 0; j < l; ++j) {
                float val = uniform(0.0f, prob[N-1]);
                distinct_ids[j] = std::lower_bound(prob, prob+N, val) - prob;
            }
        }
        // broadcast l distinct_ids from root to other threads if multi-thread case
        if (size > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(distinct_ids, l, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    // -------------------------------------------------------------------------
    //  get the global seedset and seedpos by the (t*l+1) distinct ids
    // -------------------------------------------------------------------------
    get_k_seeds<DType>(rank, size, n, avg_d, t*l+1, all_distinct_ids, dataset, 
        datawgt, datapos, datasum, seedset, seedwgt, seedpos, seedsum);

    // release space
    std::vector<int>().swap(last_seedset);
    std::vector<f32>().swap(last_seedwgt);
    std::vector<u64>().swap(last_seedpos);
    std::vector<f32>().swap(last_seedsum);
    delete[] nn_dist;
    delete[] prob;
}

// -----------------------------------------------------------------------------
void labels_to_weights(             // convert local labels to global weights
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *labels,                  // labels for n local data
    int   *weights);                    // weights for k seeds (return)

// -----------------------------------------------------------------------------
template<class DType>
void get_weights_for_seeds(         // get weights for k seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    int   *weights)                     // weights for k seeds (return)
{
    // conduct data assginment to get the local labels
    int *labels = new int[n];
    exact_assign_data<DType>(rank, n, k, dataset, datawgt, datapos, datasum,
        seedset, seedwgt, seedpos, seedsum, labels);
    
    // convert local labels into global weights
    labels_to_weights(rank, size, n, k, labels, weights);
    
    // release space
    delete[] labels;
}

// -----------------------------------------------------------------------------
template<class DType>
void kmeansll_seeding(              // init k centers by k-means||
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    int   l,                            // oversampling factor
    int   t,                            // number of iterations
    int   hard_device,                  // used hard device: 0-OpenMP, 1-GPU
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    int   *distinct_ids,                // k distinct ids (return)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // -------------------------------------------------------------------------
    //  k-means|| overseeding
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: k-means|| overseeding (t=%d, l=%d)\n\n", rank, t, l);
#endif
    int K = t * l + 1; // expected number of seeds for overseeding
    int *all_distinct_ids = new int[K];
    
    std::vector<int> over_seedset;
    std::vector<f32> over_seedwgt;
    std::vector<u64> over_seedpos;
    std::vector<f32> over_seedsum;
    
    kmeansll_overseeding<DType>(rank, size, n, N, avg_d, l, t, hard_device,
        dataset, datawgt, datapos, datasum, all_distinct_ids, over_seedset, 
        over_seedwgt, over_seedpos, over_seedsum);
    assert(over_seedpos.size() == K+1);
    assert(over_seedsum.size() == K);
    
    // -------------------------------------------------------------------------
    //  get weights for K seeds by data assignment
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: data assignment => weights (n=%d, K=%d)\n\n", rank, n, K);
#endif
    int *weights = new int[K];
    get_weights_for_seeds<DType>(rank, size, n, K, dataset, datawgt, datapos, 
        datasum, over_seedset.data(), over_seedwgt.data(), over_seedpos.data(), 
        over_seedsum.data(), weights);
    
    // -------------------------------------------------------------------------
    //  call k-means++ for final refinement using a single thread (@root)
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: k-means++ refinement (k=%d)\n\n", rank, k);
#endif
    if (rank == 0) {
        // use OpenMP for k-means++ by default
        int *local_distinct_ids = new int[k];
        kmeanspp_seeding<int>(0, 1, K, K, avg_d, k, 0, over_seedset.data(),
            over_seedwgt.data(), over_seedpos.data(), over_seedsum.data(), 
            weights, local_distinct_ids, seedset, seedwgt, seedpos, seedsum);
    
        // get distinct_ids from all_distinct_ids and local_distinct_ids
        for (int i = 0; i < k; ++i) {
            int pos = local_distinct_ids[i];
            distinct_ids[i] = all_distinct_ids[pos];
        }
        delete[] local_distinct_ids;
    }
    // broadcast distinct_ids and seeds from root to other threads
    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(distinct_ids, k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        broadcast_seeds(rank, size, seedset, seedwgt, seedpos, seedsum);
    }
    // release space
    std::vector<int>().swap(over_seedset);
    std::vector<f32>().swap(over_seedwgt);
    std::vector<u64>().swap(over_seedpos);
    std::vector<f32>().swap(over_seedsum);
    delete[] all_distinct_ids;
    delete[] weights;
}

// -----------------------------------------------------------------------------
template<class DType>
int silk_overseeding(               // silk overseeding
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   d,                            // real dimension of sparse data
    int   avg_d,                        // average dimension of sparse data
    int   m1,                           // #hash tables (1st level)
    int   h1,                           // #concat hash func (1st level)
    int   t,                            // threshold of #point IDs in a bucket
    int   m2,                           // #hash tables (2nd level)
    int   h2,                           // #concat hash func (2nd level)
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const char  *prefix,                // prefix of dataset
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // -------------------------------------------------------------------------
    //  phase 1: transform local sparse data to local buckets
    // -------------------------------------------------------------------------
    std::vector<int> bktset; // bucket set
    std::vector<u64> bktpos; // bucket position
    
    assert(h1 % 2 == 0 && h1 <= 100);
    int k = h1 / 2;
    int l = (int) ceil(sqrt(2.0f*m1)) + 1;
    int actual_d = -1;
    if (d <= 1000000) {
        // for the datasets News20, RCV1, Criteo10M, and Avazu
        actual_d = d; // reuse the original dimension
    }
    else {
        // for the datasets URL and KDD2012
        actual_d = 1000000; // use the dimension after rehashing 
    }
    assert((u64) actual_d*l*k <= MAX_WMH_SIZE && l*k <= MAX_NUM_HASH);
    int tot_size = actual_d*l*k;
    
    float *r_k = new float[tot_size];
    float *c_k = new float[tot_size];
    float *b_k = new float[tot_size];
    
    FILE  *fp  = nullptr;
    fp = fopen("../params/r.bin", "rb"); fread(r_k, sizeof(float), tot_size, fp);
    fclose(fp);
    fp = fopen("../params/c.bin", "rb"); fread(c_k, sizeof(float), tot_size, fp);
    fclose(fp);
    fp = fopen("../params/b.bin", "rb"); fread(b_k, sizeof(float), tot_size, fp);
    fclose(fp);
    
    // using weighted_minhash to get local hash results
    int *hash_results = new int[(u64) m1*n];
    int prime = 120001037 > N ? 120001037 : 1645333507;
    printf("Rank #%d: prime=%d (for data)\n", rank, prime);
    
    if (d <= 1000000) {
        // for the datasets News20, RCV1, Criteo10M, and Avazu
        weighted_minhash<DType>(rank, n, actual_d, prime, m1, l, k, r_k, c_k, b_k, 
            dataset, datawgt, datapos, hash_results);
    }
    else {
        // for the datasets URL and KDD2012
        // use alternative dataset after rehashing for two-level minhash
        char fname[200]; sprintf(fname, "%s_Rehashing_%d.bin", prefix, rank);
        fp = fopen(fname, "rb");
        if (!fp) { printf("Rank #%d: cannot open %s\n", rank, fname); exit(1); }
    
        // read new_datapos, new_dataset, and new_datawgt
        u64   *new_datapos = new u64[n+1]; fread(new_datapos, sizeof(u64), n+1, fp);
        u64   N = new_datapos[n];
        DType *new_dataset = new DType[N]; fread(new_dataset, sizeof(DType), N, fp);
        float *new_datawgt = new float[N]; fread(new_datawgt, sizeof(float), N, fp);
        fclose(fp);
        
#ifdef DEBUG_INFO
        printf("Rank #%d: using rehashing results (d=%d)\n", rank, actual_d);
#endif
        weighted_minhash<DType>(rank, n, actual_d, prime, m1, l, k, r_k, c_k, b_k, 
            new_dataset, new_datawgt, new_datapos, hash_results);
            
        delete[] new_datapos;
        delete[] new_dataset;
        delete[] new_datawgt;
    }
    
    // shift local hash results in the distributed setting
    if (size > 1) hash_results = shift_hash_results(size, n, m1, hash_results);
    
    // convert local hash results into local buckets
    int num_buckets = hash_results_to_buckets(rank, size, n, m1, t, hash_results, 
        bktset, bktpos);
    
    // get total number of buckets
    int tot_buckets = get_total_buckets(size, num_buckets);
    delete[] hash_results;
    delete[] r_k;
    delete[] c_k;
    delete[] b_k;
    
    printf("Rank #%d: num_buckets=%d, tot_buckets=%d\n\n", rank, num_buckets,
        tot_buckets);
    
    // -------------------------------------------------------------------------
    //  phase 2: transform local buckets to global bins
    // -------------------------------------------------------------------------
    std::vector<int> binset; // bin set
    std::vector<u64> binpos; // bin position
    
    int n_prime = 120001037 > tot_buckets ? 120001037 : 1645333507;
    int d_prime = 120001037 > N ? 120001037 : 1645333507;
    
    // minhash: convert local buckets into local signatures
    int *signatures = new int[(u64) num_buckets*m2];
    minhash<int>(rank, num_buckets, n_prime, d_prime, m2, h2, bktset.data(),
        bktpos.data(), signatures);
    
    // signatures_to_bins: convert local signatures into global bins
    int tot_bins = signatures_to_bins(rank, size, N, num_buckets, m2, b, 
        delta, false, gbeta, lbeta, bktset.data(), bktpos.data(), 
        signatures, binset, binpos);
    delete[] signatures; signatures = nullptr;

    printf("Rank #%d: tot_bins=%d\n\n", rank, tot_bins);
    
    // -------------------------------------------------------------------------
    //  phase 3: conduct minhash again to deduplicate global bins
    // -------------------------------------------------------------------------
    // convert global bins into local buckets
    tot_buckets = tot_bins;
    num_buckets = bins_to_buckets(rank, size, tot_buckets, binset, binpos, 
        bktset, bktpos);
    
    n_prime = 120001037 > tot_buckets ? 120001037 : 1645333507;
    d_prime = 120001037 > N ? 120001037 : 1645333507;
    
    // minhash: convert local buckets into local signatures
    m2 = 1; h2 = 2;
    signatures = new int[(u64) num_buckets*m2];
    minhash<int>(rank, num_buckets, n_prime, d_prime, m2, h2, bktset.data(), 
        bktpos.data(), signatures);
    
    // signatures_to_bins: convert local signatures into global bins
    tot_bins = signatures_to_bins(rank, size, N, num_buckets, m2, 0, 
        delta, true, gbeta, lbeta, bktset.data(), bktpos.data(), 
        signatures, binset, binpos);
    delete[] signatures; signatures = nullptr;
    
    printf("Rank #%d: K=%d\n\n", rank, tot_bins);
    
    // -------------------------------------------------------------------------
    //  phase 4: transform bins to seeds
    // -------------------------------------------------------------------------
    bins_to_seeds<DType>(rank, size, n, tot_bins, avg_d, galpha, lalpha, 
        dataset, datawgt, datapos, binset.data(), binpos.data(), seedset, 
        seedwgt, seedpos, seedsum);
    
    // clear space
    std::vector<int>().swap(bktset);
    std::vector<u64>().swap(bktpos);
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
    
    return tot_bins;
}

// -----------------------------------------------------------------------------
int early_stop(                     // early stop process
    std::vector<int> &over_seedset,     // over seed set (allow modify)
    std::vector<f32> &over_seedwgt,     // over seed weight (allow modify)
    std::vector<u64> &over_seedpos,     // over seed position (allow modify)
    std::vector<f32> &over_seedsum,     // over seed sum of weights (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum);         // seed sum of weights (return)
 
// -----------------------------------------------------------------------------
template<class DType>
void labels_to_weights(             // get weights from labels & update seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const int   *labels,                // labels for n local data
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum,          // seed sum of weights (return)
    int   *weights)                     // weights for k seeds (return)
{
    std::vector<int> binset; // bin set
    std::vector<u64> binpos; // bin position
    
    // convert local labels into global bins and re-number local labels
    int new_k = labels_to_index(size, n, k, labels, binset, binpos);
    assert(new_k <= k && new_k > 0);
    
    // update weights based on new_k bins
    for (int i = 0; i < new_k; ++i) {
        weights[i] = get_length(i, binpos.data());
    }
    // re-generate seeds based on global bins
    bins_to_seeds<DType>(rank, size, n, new_k, avg_d, galpha, lalpha, dataset, 
        datawgt, datapos, binset.data(), binpos.data(), seedset, seedwgt, 
        seedpos, seedsum);
    
    std::vector<int>().swap(binset);
    std::vector<u64>().swap(binpos);
}

// -----------------------------------------------------------------------------
template<class DType>
int get_weights_for_seeds(          // get weights for k seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum,          // seed sum of weights (return)
    int   *weights)                     // weights for k seeds (return)
{
    // conduct data assginment to get the local labels
    int *labels = new int[n];
    exact_assign_data<DType>(rank, n, k, dataset, datawgt, datapos, datasum,
        seedset.data(), seedwgt.data(), seedpos.data(), seedsum.data(),
        labels);
    
    // convert local labels into global weights
    labels_to_weights<DType>(rank, size, n, avg_d, k, galpha, lalpha, labels,
        dataset, datawgt, datapos, seedset, seedwgt, seedpos, seedsum, weights);
    
    // release space
    delete[] labels;
    return seedpos.size()-1;
}

// -----------------------------------------------------------------------------
template<class DType>
int silk_seeding(                   // init k centers by silk
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   N,                            // total number of data points
    int   d,                            // real dimension of sparse data
    int   avg_d,                        // average dimension of sparse data
    int   k,                            // number of seeds
    int   m1,                           // #hash tables (1st level)
    int   h1,                           // #concat hash func (1st level)
    int   t,                            // threshold of #point IDs in a bucket
    int   m2,                           // #hash tables (2nd level)
    int   h2,                           // #concat hash func (2nd level)
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const char  *prefix,                // prefix of dataset
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // -------------------------------------------------------------------------
    //  silk overseeding
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: silk overseeding (n=%d)\n\n", rank, n);
#endif
    int K = 0;
    std::vector<int> over_seedset;
    std::vector<f32> over_seedwgt;
    std::vector<u64> over_seedpos;
    std::vector<f32> over_seedsum;
    
    // get K seeds by silk
    K = silk_overseeding<DType>(rank, size, n, N, d, avg_d, m1, h1, t, m2, h2, 
        b, delta, gbeta, lbeta, galpha, lalpha, prefix, dataset, datawgt, 
        datapos, datasum, over_seedset, over_seedwgt, over_seedpos, over_seedsum);
    
    // stop early if have no larger than k seeds
    if (K <= k) {
        return early_stop(over_seedset, over_seedwgt, over_seedpos, 
            over_seedsum, seedset, seedwgt, seedpos, seedsum);
    }

    // -------------------------------------------------------------------------
    //  get weights for K seeds by data assignment
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: data assignment => weights (n=%d, K=%d)\n\n", rank, n, K);
#endif
    int *weights = new int[K];
    
    // get weights for K seeds (remove seeds if they have no weights)
    K = get_weights_for_seeds<DType>(rank, size, n, avg_d, K, galpha, lalpha,
        dataset, datawgt, datapos, datasum, over_seedset, over_seedwgt, 
        over_seedpos, over_seedsum, weights);
    
    // stop early if have no larger than k seeds
    if (K <= k) {
        delete[] weights;
        return early_stop(over_seedset, over_seedwgt, over_seedpos, 
            over_seedsum, seedset, seedwgt, seedpos, seedsum);
    }
    
    // -------------------------------------------------------------------------
    //  call k-means++ for final refinement using a single thread (@root)
    // -------------------------------------------------------------------------
#ifdef DEBUG_INFO
    printf("Rank #%d: k-means++ refinement (n=%d, k=%d)\n\n", rank, K, k);
#endif
    if (rank == 0) {
        // use OpenMP for k-means++ by default
        int *local_distinct_ids = new int[k];
        kmeanspp_seeding<int>(0, 1, K, K, avg_d, k, 0, over_seedset.data(),
            over_seedwgt.data(), over_seedpos.data(), over_seedsum.data(), 
            weights, local_distinct_ids, seedset, seedwgt, seedpos, seedsum);
        delete[] local_distinct_ids;
    }
    // broadcast seeds from root to other threads
    if (size > 1) {
        broadcast_seeds(rank, size, seedset, seedwgt, seedpos, seedsum);
    }
    
    // release space
    delete[] weights;
    std::vector<int>().swap(over_seedset);
    std::vector<f32>().swap(over_seedwgt);
    std::vector<u64>().swap(over_seedpos);
    std::vector<f32>().swap(over_seedsum);
    
    return k;
}

} // end namespace clustering
