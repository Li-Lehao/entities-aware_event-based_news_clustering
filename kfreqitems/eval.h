#pragma once

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

#include "def.h"
#include "util.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
template<class DType>
float calc_weighted_jaccard_dist(   // calc weighted jaccard dist of data & seed
    int   did,                          // data id
    int   sid,                          // label (seed id)
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum)               // seed sum of weights
{
    int   n_data   = get_length(did, datapos);
    float data_sum = datasum[did];
    const DType *data_coord = dataset + datapos[did];
    const float *data_weigh = datawgt + datapos[did];
    
    int   n_seed   = get_length(sid, seedpos);
    float seed_sum = seedsum[sid];
    const int   *seed_coord = seedset + seedpos[sid];
    const float *seed_weigh = seedwgt + seedpos[sid];
    
    return weighted_jaccard_dist<DType>(n_data, n_seed, data_sum, seed_sum, 
        data_coord, data_weigh, seed_coord, seed_weigh);
}

// -----------------------------------------------------------------------------
template<class DType>
void calc_local_stat_by_seeds(      // calc local statistics by seeds
    int   n,                            // number of data points
    int   k,                            // number of clusters
    const int   *labels,                // cluster labels for data points
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    float *c_mae,                       // mean absolute error (return)
    float *c_mse)                       // mean square   error (return)
{
    // calc the jaccard distance for local data to its nearest seed
    float *dist = new float[n];
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        dist[i] = calc_weighted_jaccard_dist<DType>(i, labels[i], dataset, 
            datawgt, datapos, datasum, seedset, seedwgt, seedpos, seedsum);
    }
    
    // update the statistics information
    int   sid = -1;
    float dis = -1.0f;
    for (int i = 0; i < n; ++i) {
        sid = labels[i]; dis = dist[i];
        c_mae[sid] += dis;
        c_mse[sid] += dis*dis;
    }
    delete[] dist;
}

// -----------------------------------------------------------------------------
void calc_global_stat(              // calc global statistics
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of clusters
    float *c_mae,                       // mean absolute error (allow modify)
    float *c_mse,                       // mean square   error (allow modify)
    float &mae,                         // mean absolute error (return)
    float &mse);                        // mean square error (return)

} // end namespace clustering
