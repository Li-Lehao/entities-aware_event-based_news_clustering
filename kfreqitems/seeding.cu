#include "seeding.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
void generate_k_distinct_ids(       // generate k distinct ids
    int k,                              // k value
    int n,                              // total range
    int *distinct_ids)                  // distinct ids (return)
{
    bool *select = new bool[n]; memset(select, false, sizeof(bool)*n);
    int  id = -1;
    for (int i = 0; i < k; ++i) {
        // every time draw a distinct id uniformly at from from [0,n-1]
        do { id = uniform_u32(0, n-1); } while (select[id]);
        
        select[id] = true;
        distinct_ids[i] = id;
    }
    delete[] select;
}

// -----------------------------------------------------------------------------
void gather_all_local_seedset(      // gather all local seedset to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    const std::vector<int> &local_seedset, // local seedset
    std::vector<int> &seedset)          // seedset at root (return)
{
    // gather the length of local_seedset from different threads to root
    int len = (int) local_seedset.size(); // length of local_seedset
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root: get the total length of global seedset & resize it
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    if (rank == 0) { assert(tlen <= MAX_BUFF_SIZE); seedset.resize(tlen); }
    
    // @root: init displacements to gather all local_seedset
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    // gather local_seedset from different threads to global seedset @root
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_seedset.data(), len, MPI_INT, seedset.data(), rlen, 
        displs, MPI_INT, 0, MPI_COMM_WORLD); // 0 is root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
}

// -----------------------------------------------------------------------------
void gather_all_local_seedwgt(      // gather all local seedwgt to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    const std::vector<f32> &local_seedwgt, // local seedwgt
    std::vector<f32> &seedwgt)          // seedwgt at root (return)
{
    // gather the length of local_seedwgt from different threads to root
    int len = (int) local_seedwgt.size(); // length of local_seedwgt
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root: get the total length of global seedwgt & resize it
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    if (rank == 0) { assert(tlen <= MAX_BUFF_SIZE); seedwgt.resize(tlen); }
    
    // @root: init displacements to gather all local_seedwgt
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    // gather local_seedwgt from different threads to global seedwgt @root
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_seedwgt.data(), len, MPI_FLOAT, seedwgt.data(), rlen, 
        displs, MPI_FLOAT, 0, MPI_COMM_WORLD); // 0 is root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
}

// -----------------------------------------------------------------------------
void gather_all_local_seedpos(      // gather all local seedpos to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   k,                            // k value
    const std::vector<u64> &local_seedpos, // local seedpos
    std::vector<u64> &seedpos)          // seedpos at root (return)
{
    // gather the length of local_seedpos from different threads to root
    int len = (int) local_seedpos.size()-1; // skip the first 0
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root: get the total length of global seedpos & reinit it
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    if (rank == 0) { assert(tlen == k); seedpos.resize(k+1); seedpos[0]=0; }
    
    // @root: init displacements to gather all local_seedpos
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    // gather local_seedpos from different threads to global seedpos @root
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_seedpos.data()+1, len, MPI_UINT64_T, seedpos.data()+1, 
        rlen, displs, MPI_UINT64_T, 0, MPI_COMM_WORLD); // 0 is root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
}

// -----------------------------------------------------------------------------
void gather_all_local_seedsum(      // gather all local seedsum to root
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   k,                            // k value
    const std::vector<f32> &local_seedsum, // local seedsum
    std::vector<f32> &seedsum)          // seedsum at root (return)
{
    // gather the length of local_seedsum from different threads to root
    int len = (int) local_seedsum.size(); // length of local_seedsum
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root: get the total length of global seedsum & resize it
    int tlen = 0;
    for (int i = 0; i < size; ++i) tlen += rlen[i];
    if (rank == 0) { assert(tlen == k); seedsum.resize(k); }
    
    // @root: init displacements to gather all local_seedsum
    int *displs = new int[size];
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + rlen[i-1];

    // gather local_seedsum from different threads to global seedsum @root
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_seedsum.data(), len, MPI_FLOAT, seedsum.data(), rlen, 
        displs, MPI_FLOAT, 0, MPI_COMM_WORLD); // 0 is root location
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
    delete[] displs;
}

// -----------------------------------------------------------------------------
void broadcast_target_data(         // broadcast target data to all threads
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    std::vector<int> &target_coord,     // target coord (return)
    std::vector<f32> &target_weigh,     // target weigh (return)
    float &target_sum)                  // target sum   (return)
{
    // gather the length of target data from diff threads to @root (rank=0)
    int len = (int) target_coord.size(); // length of target data
    int *rlen = new int[size];
    for (int i = 0; i < size; ++i) rlen[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&len, 1, MPI_INT, rlen, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // @root (rank=0): determine the rank of target data as new_root
    int new_root = 0;
    for (int i = 0; i < size; ++i) if (rlen[i] > 0) new_root = i;
    
    // 1. broadcast the new_root from old root (rank=0) to all threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&new_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 2. broadcast len from new_root to all threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&len, 1, MPI_INT, new_root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 3. broadcast target_coord from new_root to all threads
    if (rank != new_root) target_coord.resize(len);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(target_coord.data(), len, MPI_INT, new_root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 4. broadcast target_weigh from new_root to all threads
    if (rank != new_root) target_weigh.resize(len);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(target_weigh.data(), len, MPI_FLOAT, new_root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 4. broadcast target_sum from new_root to all threads
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&target_sum, 1, MPI_FLOAT, new_root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    delete[] rlen;
}

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_weighted_jaccard_dist(// calc weighted jaccard distance
    int   batch,                        // batch number of data points
    int   n_seed,                       // length of seed
    float seed_sum,                     // seed sum of weights
    const DType *d_dset,                // data set
    const float *d_dwgt,                // data weight
    const u64   *d_dpos,                // data position
    const float *d_dsum,                // data sum of weights
    const int   *d_seed_coord,          // seed coordinate
    const float *d_seed_weigh,          // seed weight
    float *d_dist)                      // jaccard distance (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        int   n_data   = get_length(tid, d_dpos);
        float data_sum = d_dsum[tid];
        const DType *data_coord = d_dset + d_dpos[tid];
        const float *data_weigh = d_dwgt + d_dpos[tid];
        
        float dist = weighted_jaccard_dist<DType>(n_data, n_seed, data_sum,
            seed_sum, data_coord, data_weigh, d_seed_coord, d_seed_weigh);
        
        if (d_dist[tid] > dist) d_dist[tid] = dist;
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
    float *nn_dist)                     // nn_dist (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);
    
    // declare parameters and allocation
    int n_seed = (int) seed_coord.size();
    int *d_seed_coord; cudaMalloc((void**)&d_seed_coord, sizeof(int)*n_seed);
    f32 *d_seed_weigh; cudaMalloc((void**)&d_seed_weigh, sizeof(f32)*n_seed);
    
    cudaMemcpy(d_seed_coord, seed_coord.data(), sizeof(int)*n_seed, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed_weigh, seed_weigh.data(), sizeof(f32)*n_seed, cudaMemcpyHostToDevice);
    
    // mem_avail = total_mem - memory(d_seed_coord + d_seed_weigh)
    u64 mem_avail = GPU_MEMORY_LIMIT - (sizeof(int)+sizeof(float))*n_seed;
    u64 mem_usage = 0UL, n_dset = 0UL;
    int n_dpos = 0, batch = 0, start = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dwgt + d_dpos + d_dsum + d_dist
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = (sizeof(DType) + sizeof(float))*n_dset + 
                sizeof(u64)*n_dpos + sizeof(float)*(batch+1)*2;
        }
        // ---------------------------------------------------------------------
        //  parallel nn_dist update for batch data if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = (sizeof(DType) + sizeof(float))*n_dset + 
                sizeof(u64)*n_dpos + sizeof(float)*batch*2;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU 
            float *h_dist = nn_dist + start; // nn_dist @host (allow modify)
            const DType *h_dset = dataset + datapos[start]; // data set @host
            const float *h_dwgt = datawgt + datapos[start]; // data weight @host
            const float *h_dsum = datasum + start;          // data sum @host
            u64   *h_dpos = new u64[n_dpos];                // data position @host
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset; cudaMalloc((void**) &d_dset, sizeof(DType)*n_dset);
            float *d_dwgt; cudaMalloc((void**) &d_dwgt, sizeof(float)*n_dset);
            u64   *d_dpos; cudaMalloc((void**) &d_dpos, sizeof(u64)  *n_dpos);
            float *d_dsum; cudaMalloc((void**) &d_dsum, sizeof(float)*batch);
            float *d_dist; cudaMalloc((void**) &d_dist, sizeof(float)*batch);
            
            cudaMemcpy(d_dset, h_dset, sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dwgt, h_dwgt, sizeof(float)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos, h_dpos, sizeof(u64)  *n_dpos, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dsum, h_dsum, sizeof(float)*batch,  cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist, h_dist, sizeof(float)*batch,  cudaMemcpyHostToDevice);
            
            // calc Jaccard distance for between batch data and seed
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch + block-1) / block;
            calc_weighted_jaccard_dist<DType><<<grid, block>>>(batch, n_seed,
                seed_sum, d_dset, d_dwgt, d_dpos, d_dsum, d_seed_coord, 
                d_seed_weigh, d_dist);
            
            // update nn_dist for batch data & release local space
            cudaMemcpy(h_dist, d_dist, sizeof(float)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dwgt); cudaFree(d_dpos); cudaFree(d_dsum);
            cudaFree(d_dist);
            delete[] h_dpos;
            
            // update local parameters for next nn_dist update
            start += batch; batch = 0;
        }
        if (i < n) ++batch;
    }
    assert(start == n);
    cudaFree(d_seed_coord); cudaFree(d_seed_weigh);
}

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const u08   *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seed_coord, // last seed coord
    const std::vector<f32> &seed_weigh, // last seed weigh
    float seed_sum,                     // last seed sum of weights
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const u16   *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seed_coord, // last seed coord
    const std::vector<f32> &seed_weigh, // last seed weigh
    float seed_sum,                     // last seed sum of weights
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    const int   *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    const float *datasum,               // data sum of weights
    const std::vector<int> &seed_coord, // last seed coord
    const std::vector<f32> &seed_weigh, // last seed weigh
    float seed_sum,                     // last seed sum of weights
    float *nn_dist);                    // nn_dist (return)

// -----------------------------------------------------------------------------
template<class DType>
__global__ void calc_weighted_jaccard_dist(  // calc weighted jaccard distance
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
    float *d_dist)                      // jaccard distance (return)
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
__global__ void update_nn_dist(     // update nn_dist for batch data
    int   batch,                        // batch number of data points
    int   k,                            // number of seeds
    const float *d_dist,                // jaccard distance array
    float *d_nn_dist)                   // nn_dist (return)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch) {
        const float *dist = d_dist + (u64) tid*k; // get dist array
        float min_dist = d_nn_dist[tid];
        
        // update min_dist among the k dist
        for (int i = 0; i < k; ++i) {
            if (dist[i] < min_dist) min_dist = dist[i];
        }
        d_nn_dist[tid] = min_dist;
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
    float *nn_dist)                     // nn_dist (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);

    // declare parameters and allocation
    u64 len = seedpos[k];
    int *d_sset = nullptr; cudaMalloc((void**)&d_sset, sizeof(int)*len);
    f32 *d_swgt = nullptr; cudaMalloc((void**)&d_swgt, sizeof(f32)*len);
    u64 *d_spos = nullptr; cudaMalloc((void**)&d_spos, sizeof(u64)*(k+1));
    f32 *d_ssum = nullptr; cudaMalloc((void**)&d_ssum, sizeof(f32)*k);
    
    cudaMemcpy(d_sset, seedset, sizeof(int)*len,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_swgt, seedwgt, sizeof(f32)*len,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_spos, seedpos, sizeof(u64)*(k+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ssum, seedsum, sizeof(f32)*k,     cudaMemcpyHostToDevice);

    // mem_avail = total_mem - memory(d_sset + d_swgt + d_spos)
    u64 mem_avail = GPU_MEMORY_LIMIT - (sizeof(int)*len + sizeof(f32)*len + 
        sizeof(u64)*(k+1) + sizeof(f32)*k);
    u64 mem_usage = 0UL, n_dset = 0UL;
    int n_dpos = 0, batch = 0, start  = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_dset + d_dwgt + d_dpos + d_dsum + d_dist + d_nn_dist
            n_dset = datapos[i+1] - datapos[start];
            n_dpos = batch + 2;
            mem_usage = (sizeof(DType) + sizeof(float))*n_dset + 
                sizeof(u64)*n_dpos + sizeof(float)*(batch+1) + 
                sizeof(float)*(k+1)*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel batch data assignment if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            n_dset = datapos[i] - datapos[start];
            n_dpos = batch + 1;
            mem_usage = (sizeof(DType) + sizeof(float))*n_dset + 
                sizeof(u64)*n_dpos + sizeof(float)*batch + 
                sizeof(float)*(k+1)*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n",
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU
            float *h_nn_dist = nn_dist + start; // allow modify
            const DType *h_dset = dataset + datapos[start]; // data set @host
            const float *h_dwgt = datawgt + datapos[start]; // data weight @host
            const float *h_dsum = datasum + start;          // data sum @host
            u64   *h_dpos = new u64[n_dpos];                // data pos @host
            copy_pos(n_dpos, datapos + start, h_dpos);
            
            DType *d_dset;    cudaMalloc((void**) &d_dset,    sizeof(DType)*n_dset);
            float *d_dwgt;    cudaMalloc((void**) &d_dwgt,    sizeof(float)*n_dset);
            u64   *d_dpos;    cudaMalloc((void**) &d_dpos,    sizeof(u64)  *n_dpos);
            float *d_dsum;    cudaMalloc((void**) &d_dsum,    sizeof(float)*batch);
            float *d_dist;    cudaMalloc((void**) &d_dist,    sizeof(float)*batch*k);
            float *d_nn_dist; cudaMalloc((void**) &d_nn_dist, sizeof(float)*batch);
            
            cudaMemcpy(d_dset,    h_dset,    sizeof(DType)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dwgt,    h_dwgt,    sizeof(float)*n_dset, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dpos,    h_dpos,    sizeof(u64)  *n_dpos, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dsum,    h_dsum,    sizeof(float)*batch,  cudaMemcpyHostToDevice);
            cudaMemcpy(d_nn_dist, h_nn_dist, sizeof(float)*batch,  cudaMemcpyHostToDevice);
            
            // calc Jaccard distance between batch data and k seeds
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch*k + block-1) / block;
            calc_weighted_jaccard_dist<DType><<<grid, block>>>(batch, k, d_dset, 
                d_dwgt, d_dpos, d_dsum, d_sset, d_swgt, d_spos, d_ssum, d_dist);
            
            // update the nn_dist for batch data
            grid = ((u64) batch + block-1) / block;
            update_nn_dist<<<grid, block>>>(batch, k, d_dist, d_nn_dist);
            
            // get the new nn_dist & release local space
            cudaMemcpy(h_nn_dist, d_nn_dist, sizeof(float)*batch, cudaMemcpyDeviceToHost);
            
            cudaFree(d_dset); cudaFree(d_dwgt); cudaFree(d_dpos); cudaFree(d_dsum);
            cudaFree(d_dist); cudaFree(d_nn_dist);
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
template void update_nn_dist(       // update nn_dist for local data by k seeds
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
    float *nn_dist);                    // nn_dist (return)
    
// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data by k seeds
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
    float *nn_dist);                    // nn_dist (return)
    
// -----------------------------------------------------------------------------
template void update_nn_dist(       // update nn_dist for local data by k seeds
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
    float *nn_dist);                    // nn_dist (return)
    
// -----------------------------------------------------------------------------
void labels_to_weights(             // convert local labels to global weights
    int   rank,                         // MPI rank
    int   size,                         // number of MPIs (size)
    int   n,                            // number of data points
    int   k,                            // number of seeds
    const int *labels,                  // labels for n local data
    int   *weights)                     // weights for k seeds (return)
{
    assert((u64) n*size < MAX_INT); // total num of data points
    int N = n*size; 
    int *all_labels = new int[N];
    
    // -------------------------------------------------------------------------
    //  get all labels from different threads to root
    // -------------------------------------------------------------------------
    if (size == 1) {
        // directly copy labels to all labels
        std::copy(labels, labels + n, all_labels);
    }
    else {
        // get all labels from different threads to root
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(labels, n, MPI_INT, all_labels, n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // -------------------------------------------------------------------------
    //  @root: sequentical counting the number of labels as weight for each seed
    // -------------------------------------------------------------------------
    if (rank == 0) {
        memset(weights, 0, sizeof(int)*k); // init weights
        int pos = -1;
        for (int i = 0; i < N; ++i) { pos = all_labels[i]; ++weights[pos]; }
    }
    delete[] all_labels;
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
    std::vector<f32> &seedsum)          // seed sum of weights (return)
{
    // clear original space for seedset & seedpos
    std::vector<int>().swap(seedset);
    std::vector<f32>().swap(seedwgt);
    std::vector<u64>().swap(seedpos);
    std::vector<f32>().swap(seedsum);
    
    // swap contents for over_seedset and seedset
    seedset.swap(over_seedset);
    seedwgt.swap(over_seedwgt);
    seedpos.swap(over_seedpos);
    seedsum.swap(over_seedsum);
    
    return seedsum.size();
}

} // end namespace clustering
