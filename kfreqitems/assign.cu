#include "assign.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
struct TripletCmp {
    __host__ __device__
    bool operator() (const Triplet &t1, const Triplet &t2)
    {
        return t1.coord_ < t2.coord_;
    }
};

// -----------------------------------------------------------------------------
void frequent_items(                // apply frequent items for cand_list
    float galpha,                       // global alpha
    std::vector<Triplet> &cand_list,    // cand_list (allow modify)
    std::vector<int> &seedset,          // seed set (return)
    std::vector<f32> &seedwgt,          // seed weight (return)
    std::vector<u64> &seedpos,          // seed position (return)
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // first sort cand_list in ascending order by PairCmp
    thrust::sort(cand_list.begin(), cand_list.end(), TripletCmp());
    
    // get the distinct coordinates and sum of weights and frequencies
    u64 total_num = cand_list.size();
    int *coord = new int[total_num];
    f32 *weigh = new f32[total_num];
    int *freq  = new int[total_num];
    
    int max_freq = 0; 
    f32 max_weight = 0.0f;
    int num = 0; freq[0] = 0; weigh[0] = 0.0f;
    for (size_t i = 1; i < total_num; ++i) {
        freq[num]  += cand_list[i-1].freq_;
        weigh[num] += cand_list[i-1].weigh_;
        
        if (cand_list[i].coord_ > cand_list[i-1].coord_) {
            coord[num] = cand_list[i-1].coord_;
            if (freq[num]  > max_freq)   max_freq   = freq[num];
            if (weigh[num] > max_weight) max_weight = weigh[num];
            
            ++num; freq[num] = 0; weigh[num] = 0.0f;
        }
    }
    freq[num]  += cand_list[total_num-1].freq_;
    weigh[num] += cand_list[total_num-1].weigh_;
    coord[num]  = cand_list[total_num-1].coord_;
    if (freq[num]  > max_freq)   max_freq   = freq[num];
    if (weigh[num] > max_weight) max_weight = weigh[num];
    ++num;
    
    // select the coords with high frequency as seeds
    int   *seed_coord = new int[num];
    float *seed_weigh = new float[num];
    float sum = 0.0f;
    int   cnt = 0; // number of coordinates for seed
    if (max_freq == 1) {
        // further consider weight for filtering
        float threshold = max_weight*galpha;
        for (int i = 0; i < num; ++i) {
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
        for (int i = 0; i < num; ++i) {
            if (freq[i] >= threshold) { 
                seed_coord[cnt] = coord[i]; 
                seed_weigh[cnt] = weigh[i] / freq[i]; // average weight by freq
                sum += seed_weigh[cnt];
                ++cnt;
            }
        }
    }
    // update seedset, seedwgt, seedpos, and seedsum
    seedset.insert(seedset.end(), seed_coord, seed_coord+cnt);
    seedwgt.insert(seedwgt.end(), seed_weigh, seed_weigh+cnt);
    seedpos.push_back(cnt); // add cnt to seedpos
    seedsum.push_back(sum);
    
    // release space
    std::vector<Triplet>().swap(cand_list);
    delete[] coord; delete[] weigh; delete[] freq;
    delete[] seed_coord; delete[] seed_weigh;
}

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_jaccard_dist(  // calc jaccard distance
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const DType *d_dset,                // data set
    const u64   *d_dpos,                // data position
    const int   *d_sset,                // seed set
    const u64   *d_spos,                // seed position
    float *d_dist)                      // jaccard distance (return)
{
    u64 tid = (u64) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (u64) batch * k) {
        const u64   *dpos = d_dpos + (u64) tid/k;
        const DType *data = d_dset + dpos[0];
        int n_data = get_length(0, dpos);
        
        const u64 *spos = d_spos + (u64) tid%k;
        const int *seed = d_sset + spos[0];
        int n_seed = get_length(0, spos);
        
        d_dist[tid] = jaccard_dist<DType>(n_data, n_seed, data, seed);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_weighted_jaccard_dist(// calc weighted jaccard distance
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const DType *d_dset,                // data set
    const float *d_dwgt,                // data weight
    const u64   *d_dpos,                // data position
    const float *d_dsum,                // data sum of weights
    const int   *d_sset,                // seed set
    const float *d_swgt,                // seed weight
    const u64   *d_spos,                // seed position
    const float *d_ssum,                // seed sum of weights
    float *d_dist)                      // weighted jaccard distance (return)
{
    u64 tid = (u64) blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < (u64) batch * k) {
        u64   did      = tid / k; // data id
        int   n_data   = get_length(did, d_dpos);
        float data_sum = d_dsum[did];
        const DType *data_coord = d_dset + d_dpos[did];
        const float *data_weigh = d_dwgt + d_dpos[did];
        
        u64   sid      = tid % k; // seed id
        int   n_seed   = get_length(sid, d_spos);
        float seed_sum = d_ssum[sid];
        const int   *seed_coord = d_sset + d_spos[sid];
        const float *seed_weigh = d_swgt + d_spos[sid];
        
        d_dist[tid] = weighted_jaccard_dist<DType>(n_data, n_seed, data_sum, 
            seed_sum, data_coord, data_weigh, seed_coord, seed_weigh);
    }
}

// -----------------------------------------------------------------------------
__global__ void nearest_seed(       // find the nearest seed id for batch data
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const float *d_dist,                // jaccard distance array
    int   *d_labels)                    // labels (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        const float *dist = d_dist + (u64) tid*k; // get dist array
        
        // find the minimum jaccard distance for batch data
        int   min_id   = 0;
        float min_dist = dist[0];
        for (int i = 1; i < k; ++i) {
            if (dist[i] < min_dist) { min_id = i; min_dist = dist[i]; }
        }
        d_labels[tid] = min_id;
    }
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
    int   *labels)                      // cluster labels for dataset (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);

    // declare parameters and allocation
    u64   len = seedpos[k];
    int   *d_sset; cudaMalloc((void**)&d_sset, sizeof(int)*len);
    float *d_swgt; cudaMalloc((void**)&d_swgt, sizeof(float)*len);
    u64   *d_spos; cudaMalloc((void**)&d_spos, sizeof(u64)*(k+1));
    float *d_ssum; cudaMalloc((void**)&d_ssum, sizeof(float)*k);
    
    cudaMemcpy(d_sset, seedset, sizeof(int)*len,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_swgt, seedwgt, sizeof(float)*len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spos, seedpos, sizeof(u64)*(k+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ssum, seedsum, sizeof(float)*k,   cudaMemcpyHostToDevice);

    // mem_avail = total_mem - memory(d_sset + d_swgt + d_spos + d_ssum)
    u64 mem_avail = GPU_MEMORY_LIMIT - (sizeof(int)*len + sizeof(float)*len + 
        sizeof(u64)*(k+1) + sizeof(float)*k);
    u64 mem_usage = 0UL, n_dset = 0UL;
    int n_dpos = 0, batch = 0, start = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dwgt + d_dpos + d_dsum + d_dist + d_labels
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = (sizeof(DType) + sizeof(float))*n_dset + 
                sizeof(u64)*n_dpos + sizeof(float)*(batch+1) + 
                (sizeof(float)*k + sizeof(int))*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel batch data assignment if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = (sizeof(DType) + sizeof(float))*n_dset + 
                sizeof(u64)*n_dpos + sizeof(float)*batch + 
                (sizeof(float)*k + sizeof(int))*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU
            const DType *h_dset = dataset + datapos[start]; // data set @host
            const float *h_dwgt = datawgt + datapos[start]; // data weight @host
            const float *h_dsum = datasum + start;          // data sum @host
            u64   *h_dpos = new u64[n_dpos];                // data position @host
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset;   cudaMalloc((void**) &d_dset,   sizeof(DType)*n_dset);
            float *d_dwgt;   cudaMalloc((void**) &d_dwgt,   sizeof(float)*n_dset);
            u64   *d_dpos;   cudaMalloc((void**) &d_dpos,   sizeof(u64)*n_dpos);
            float *d_dsum;   cudaMalloc((void**) &d_dsum,   sizeof(float)*batch);
            float *d_dist;   cudaMalloc((void**) &d_dist,   sizeof(float)*k*batch);
            int   *d_labels; cudaMalloc((void**) &d_labels, sizeof(int)*batch);
            
            cudaMemcpy(d_dset, h_dset, sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dwgt, h_dwgt, sizeof(float)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos, h_dpos, sizeof(u64)*n_dpos,   cudaMemcpyHostToDevice);
            cudaMemcpy(d_dsum, h_dsum, sizeof(float)*batch,  cudaMemcpyHostToDevice);
            
            // compute weighted Jaccard distance for between batch data and k seeds
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch*k + block-1) / block;
            calc_weighted_jaccard_dist<DType><<<grid, block>>>(batch, k, d_dset, 
                d_dwgt, d_dpos, d_dsum, d_sset, d_swgt, d_spos, d_ssum, d_dist);
            
            // find the nearest seed for batch data
            grid = ((u64) batch + block-1) / block;
            nearest_seed<<<grid, block>>>(batch, k, d_dist, d_labels);
            
            // update labels & release local space
            cudaMemcpy(&labels[start], d_labels, sizeof(int)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dwgt); cudaFree(d_dpos); cudaFree(d_dsum);
            cudaFree(d_dist); cudaFree(d_labels);
            delete[] h_dpos;
            
            // update local parameters for next batch data assignment
            start += batch; batch = 0;
        }
        if (i < n) ++batch;
    }
    assert(start == n);
    cudaFree(d_sset); cudaFree(d_swgt); cudaFree(d_spos); cudaFree(d_ssum);
}

// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u08   *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const u16   *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    int   *labels);                     // cluster labels for data (return)
    
// -----------------------------------------------------------------------------
template void exact_assign_data(    // exact sparse data assginment
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int   *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const int   *seedset,               // seed set
    const float *seedwgt,               // seed weight
    const u64   *seedpos,               // seed position
    const float *seedsum,               // seed sum of weights
    int   *labels);                     // cluster labels for data (return)
    
} // end namespace clustering
