#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
template<class DType>
void global_frequent_items(         // apply frequent items for global data
    int   num,                          // number of point IDs in a bin
    float galpha,                       // global alpha
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const int   *bin,                   // bin
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // get total number of coordinates (total_num) in this bin
    u64 total_num = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; total_num += datapos[id+1] - datapos[id];
    }
    
    // init two arrays to store all coordinates and weights in this bin
    DType *arr_coord = new DType[total_num];
    float *arr_weigh = new float[total_num];
    u64   cnt = 0UL;
    for (int i = 0; i < num; ++i) {
        // copy the coordinates and weights of this data to the array
        const u64 *pos = datapos + bin[i];     // get data pos
        int   len = get_length(0, pos);        // get data len
        
        const DType *coord = dataset + pos[0]; // get coordinates
        const float *weigh = datawgt + pos[0]; // get weights
        std::copy(coord, coord+len, arr_coord+cnt);
        std::copy(weigh, weigh+len, arr_weigh+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and sum of weights and frequencies
    DType *coord = new DType[total_num]; // distinct coordinates
    float *weigh = new float[total_num]; // sum of weights
    int   *freq  = new int[total_num];   // sum of frequencies
    
    int n = 0;              // number of distinct coordinates
    int max_freq = -1;      // maximum frequency
    f32 max_weight = -1.0f; // maximum weight
    distinct_coord_weigh_freq<DType>(total_num, arr_coord, arr_weigh, 
        coord, weigh, freq, n, max_freq, max_weight);
    
    // select the coords with high frequency as seeds
    DType *seed_coord = new DType[n];
    float *seed_weigh = new float[n];
    float sum = 0.0f;
    cnt = 0; // number of coordinates for seed
    if (max_freq == 1) {
        // further consider weight for filtering
        float threshold = max_weight*galpha;
        for (int i = 0; i < n; ++i) {
            if (weigh[i] >= threshold) {
                seed_coord[cnt] = coord[i];
                seed_weigh[cnt] = weigh[i] / freq[i]; // average weight by freq
                sum += seed_weigh[cnt];
                ++cnt;
            }
        }
    }
    else {
        // directly using freq for filtering
        int threshold = (int) ceil((double) max_freq*galpha);
        for (int i = 0; i < n; ++i) {
            if (freq[i] >= threshold) {
                seed_coord[cnt] = coord[i];
                seed_weigh[cnt] = weigh[i] / freq[i]; // average weight by freq
                sum += seed_weigh[cnt];
                ++cnt;
            }
        }
    }
    // update seedset, seedwgt, and seedpos
    seedset.insert(seedset.end(), seed_coord, seed_coord+cnt);
    seedwgt.insert(seedwgt.end(), seed_weigh, seed_weigh+cnt);
    seedpos.push_back(cnt);
    seedsum.push_back(sum);
    
    // release space
    delete[] coord; delete[] weigh; delete[] freq;
    delete[] arr_coord;  delete[] arr_weigh;
    delete[] seed_coord; delete[] seed_weigh;
}

// -----------------------------------------------------------------------------
template<class DType>
void local_frequent_items(          // apply frequent items for local data
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   num,                          // number of point IDs in a bin
    int   local_max_len,                // local seed dimension
    float lalpha,                       // local alpha
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const int   *bin,                   // bin
    int   *local_coord,                 // local coordinates (return)
    float *local_weigh,                 // local weights (return)
    int   *local_freq,                  // local frequencies (return)
    int   &local_len)                   // actual length of local seed (return)
{
    // get the total number of coordinates (total_num) in this bin
    int lower_bound = n * rank;
    int upper_bound = n + lower_bound;
    u64 total_num = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; // only consider local data
        if (id < lower_bound || id >= upper_bound) continue;
        
        // sum up the total_num
        id -= lower_bound;
        total_num += datapos[id+1] - datapos[id];
    }
    if (total_num == 0) { local_len = 0; return; }
    
    // init two arrays to store all coordinates and weights in this bin
    DType *arr_coord = new DType[total_num];
    float *arr_weigh = new float[total_num];
    u64   cnt = 0UL;
    for (int i = 0; i < num; ++i) {
        int id = bin[i]; // only consider local data
        if (id < lower_bound || id >= upper_bound) continue;
        
        // copy the coordinates of this data to the array
        id -= lower_bound;                 // get data id
        int len = get_length(id, datapos); // get data len
        
        const DType *coord = dataset + datapos[id]; // get coordinates
        const float *weigh = datawgt + datapos[id]; // get weights
        std::copy(coord, coord+len, arr_coord+cnt);
        std::copy(weigh, weigh+len, arr_weigh+cnt);
        cnt += len;
    }
    assert(cnt == total_num);
    
    // get the distinct coordinates and their frequencies
    DType *coord = new DType[total_num];
    float *weigh = new float[total_num];
    int   *freq  = new int[total_num];
    
    int coord_size = 0;     // number of distinct coordinates
    int max_freq   = -1;    // maximum frequency
    f32 max_weight = -1.0f; // maximum weight
    distinct_coord_weigh_freq<DType>(total_num, arr_coord, arr_weigh, 
        coord, weigh, freq, coord_size, max_freq, max_weight);
    
    // select the coords with high frequency as local seeds
    if (max_freq == 1) {
        // further consider weight for filtering
        float threshold = max_weight*lalpha;
        local_len = 0;
        for (int i = 0; i < coord_size; ++i) {
            if (weigh[i] >= threshold) {
                local_coord[local_len] = (int) coord[i];
                local_weigh[local_len] = weigh[i];
                local_freq[local_len]  = freq[i];
                if (++local_len >= local_max_len) break;
            }
        }
    }
    else {
        int threshold = (int) ceil((double) max_freq*lalpha);
        local_len = 0;
        for (int i = 0; i < coord_size; ++i) {
            if (freq[i] >= threshold) {
                local_coord[local_len] = (int) coord[i];
                local_weigh[local_len] = weigh[i];
                local_freq[local_len]  = freq[i];
                if (++local_len >= local_max_len) break;
            }
        }
    }
    // release space
    delete[] arr_coord; delete[] arr_weigh;
    delete[] coord; delete[] weigh; delete[] freq;
}

// -----------------------------------------------------------------------------
void frequent_items(                // apply frequent items for cand_list
    float galpha,                       // global alpha
    std::vector<Triplet> &cand_list,    // cand_list (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum);         // seed sum of weights (return)

// -----------------------------------------------------------------------------
template<class DType>
void bins_to_seeds(                 // convert global bins into global seeds
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of bins (and seeds)
    int   avg_d,                        // average dimension of data points
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const int   *binset,                // bin set
    const u64   *binpos,                // bin position
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // clear seedset and seedpos
    std::vector<int>().swap(seedset);
    std::vector<f32>().swap(seedwgt);
    std::vector<u64>().swap(seedpos);
    std::vector<f32>().swap(seedsum);
    
    // estimate cost for seedset, seedwgt, and seedpos
    srand(RANDOM_SEED);            // fix a random seed
    seedset.reserve((u64)k*avg_d); // estimation, not correct
    seedwgt.reserve((u64)k*avg_d); // estimation, not correct
    seedpos.reserve(k+1);          // correct reservation
    seedsum.reserve(k);            // correct reservation
    seedpos.push_back(0UL);
    
    // -------------------------------------------------------------------------
    //  calc k local seeds
    // -------------------------------------------------------------------------
    int local_max_len = avg_d*10000; // estimation, has protection
    int *local_coord  = new int[local_max_len];
    f32 *local_weigh  = new f32[local_max_len];
    int *local_freq   = new int[local_max_len];
    
    int  max_buff = MAX_BUFF_SIZE / size;
    int *all_buff = new int[MAX_BUFF_SIZE];
    f32 *all_buff_weights = new float[MAX_BUFF_SIZE];
    
    int  j = 0; // the counter of #buffers, start from 0
    std::vector<std::vector<int> > buffer;
    std::vector<std::vector<f32> > buffer_weight;
    std::unordered_map<int, std::vector<Triplet> > cand_list;
    
    for (int i = 0; i < k; ++i) {
        const int *bin = binset + binpos[i];  // get bin
        int bin_size = get_length(i, binpos); // get bin size
        
        if (size == 1) {
            // -----------------------------------------------------------------
            //  single-thread case: apply frequent items to get seeds
            // -----------------------------------------------------------------
            global_frequent_items<DType>(bin_size, galpha, dataset, datawgt, 
                datapos, bin, seedset, seedwgt, seedpos, seedsum);
        }
        else {
            // -----------------------------------------------------------------
            //  multi-thread case: apply frequent items to get local_coord, 
            //  local_weigh, & local_freq and add them into buffer
            // -----------------------------------------------------------------
            if (i == 0) {
                buffer.push_back(std::vector<int>()); // init buffer
                buffer[j].reserve(max_buff);
                
                buffer_weight.push_back(std::vector<f32>());
                buffer_weight[j].reserve(max_buff);
            }
            int len = 0; // length for local seed (should <= local_max_len)
            local_frequent_items<DType>(rank, size, n, bin_size, local_max_len,
                lalpha, dataset, datawgt, datapos, bin, local_coord, local_weigh,
                local_freq, len);
            
            // add local_coord, local_weigh, local_freq to buffer & buffer_weight
            if (len == 0) continue; // no dimension in this seed
            if (len*2+2 > max_buff) len = max_buff/2-1; // cut tails if too many
            
            if (buffer[j].size()+len*2+2 > max_buff) {
                ++j;
                buffer.push_back(std::vector<int>());
                buffer[j].reserve(max_buff);
                
                buffer_weight.push_back(std::vector<f32>());
                buffer_weight[j].reserve(max_buff);
            }
            buffer[j].push_back(i);
            buffer[j].push_back(len);
            buffer[j].insert(buffer[j].end(), local_coord, local_coord+len);
            buffer[j].insert(buffer[j].end(), local_freq,  local_freq+len);
            
            buffer_weight[j].insert(buffer_weight[j].end(), local_weigh, 
                local_weigh+len);
        }
    }
    // -------------------------------------------------------------------------
    //  multi-thread case: calc k global seeds
    // -------------------------------------------------------------------------
    if (size > 1) {
        // get max_round based on j (local #buffers)
        int max_round = get_max_round(size, j); // start from 1
        
        for (int i = 0; i < max_round; ++i) {
            // gather all buffers from different local buffers to root
            if (j < i) {
                buffer.push_back(std::vector<int>());
                buffer_weight.push_back(std::vector<f32>());
            }
            int tlen = gather_all_buffers(size, buffer[i], all_buff);
            int wlen = gather_all_buffers(size, buffer_weight[i], all_buff_weights);
            
            // @root: convert all buffers into candidate list
            if (rank == 0) {
                all_buff_to_cand_list(tlen, wlen, all_buff, all_buff_weights, 
                    cand_list);
            }
        }
        // @root: convert candidate list into global seeds
        if (rank == 0) {
            for (int i = 0; i < k; ++i) {
                frequent_items(galpha, cand_list[i], seedset, seedwgt, seedpos, seedsum);
            }
        }
        // clear space
        std::vector<std::vector<int> >().swap(buffer);
        std::vector<std::vector<f32> >().swap(buffer_weight);
        cand_list.clear();
    }
    // release space
    delete[] local_coord; delete[] local_weigh; delete[] local_freq;
    delete[] all_buff; delete[] all_buff_weights;
    
    // broadcast global seeds from root to other threads
    if (size > 1) {
        broadcast_seeds(rank, size, seedset, seedwgt, seedpos, seedsum);
    }
    // accumulate the seed size to get the start position of each seed
    size_t n_seedpos = seedpos.size(); assert(n_seedpos == k+1);
    for (size_t i = 1; i < n_seedpos; ++i) seedpos[i] += seedpos[i-1];
    
#ifdef DEBUG_INFO
    printf("Rank #%d: avg_d=%d, total=%d\n", rank, seedpos[k]/k, seedpos[k]);
#endif
}

// -----------------------------------------------------------------------------
template<class DType>
void exact_assign_data(             // exact sparse data assginment
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
    int   *labels);                     // cluster labels for dataset (return)

} // end namespace clustering
