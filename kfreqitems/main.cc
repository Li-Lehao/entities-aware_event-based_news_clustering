#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdint.h>

#include "k_freqitems.h"
#include "silk.h"

using namespace clustering;

// -----------------------------------------------------------------------------
void usage()                        // display the usage
{
    printf("\n"
        "--------------------------------------------------------------------\n"
        " Parameters for K-FreqItems & SILK                                  \n"
        "--------------------------------------------------------------------\n"
        " -n  {integer}  number of data points in a data set\n"
        " -d  {integer}  data dimension\n"
        " -k  {integer}  number of clusters\n"
        " -m  {integer}  maximum iterations\n"
        " -s  {integer}  seeding algo (0-random 1-kmeans++ 2-kmeans|| 3-silk)\n" 
        " -m1 {integer}  #hash tables (1st level minhash)\n"
        " -h1 {integer}  #concatenated hash functions (1st level minhash)\n"
        " -t  {integer}  threshold of #point IDs in a bucket\n"
        " -m2 {integer}  #hash tables (2nd level minhash)\n"
        " -h2 {integer}  #concatenated hash funcions (2nd level minhash)\n"
        " -b  {integer}  threshold of #bucket IDs in a bin\n"
        " -D  {integer}  threshold of #point IDs in a bin\n"
        " -GB {real}     global beta\n"
        " -LB {real}     local  beta\n"
        " -GA {real}     global alpha\n"
        " -LA {real}     local  alpha\n"
        " -F  {string}   data format: uint8, uint16, int32, float32\n"
        " -P  {string}   prefix of data set\n"
        " -O  {string}   output folder to store output files\n"
        "\n\n\n");
}

// -----------------------------------------------------------------------------
template<class DType>
void silk_impl(                     // silk implementation
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   m,                            // #hash tables (data transform)
    int   h,                            // #concat hash func (data transform)
    int   t,                            // threshold of #point IDs in a bucket
    int   l,                            // #hash tables (silk)
    int   k,                            // #concat hash func (silk)
    int   b,                            // threshold of #bucket IDs in a bin
    int   delta,                        // threshold of #point IDs in a bin
    float gbeta,                        // global beta
    float lbeta,                        // local  beta
    float galpha,                       // global alpha
    float lalpha,                       // local  alpha
    const char *prefix,                 // prefix for data set
    const char *folder)                 // output folder to store output files
{
    MPI_INFO mpi_info;
    init_mpi_comm(mpi_info);
    omp_set_num_threads(THREAD_NUM);
    cudaSetDevice(DEVICE_LOCAL_RANK);
    
    int rank = mpi_info.rank_;
    int size = mpi_info.size_;

    // -------------------------------------------------------------------------
    //  read weighted dataset from bin file
    // -------------------------------------------------------------------------
    // double start_time = omp_get_wtime();
    // std::ios::sync_with_stdio(false);

    // char fname[200]; sprintf(fname, "%s_%d.bin", prefix, rank);
    // FILE *fp = fopen(fname, "rb");
    // if (!fp) { printf("Rank #%d: cannot open %s\n", rank, fname); exit(1); }

    // // read datapos, dataset, and datawgt
    // u64   *datapos = new u64[n+1]; fread(datapos, sizeof(u64), n+1, fp);
    // u64   N = datapos[n];
    // DType *dataset = new DType[N]; fread(dataset, sizeof(DType), N, fp);
    // float *datawgt = new float[N]; fread(datawgt, sizeof(float), N, fp);
    // fclose(fp);
    
    // double loading_time = omp_get_wtime() - start_time;
    // printf("Rank #%d: n=%d, d=%d, N=%lu, time=%.2lf seconds, path=%s\n\n", 
    //     rank, n, d, N, loading_time, fname);

    // -------------------------------------------------------------------------
    //  read weighted dataset from txt file
    // -------------------------------------------------------------------------
    double start_time = omp_get_wtime();
    std::ios::sync_with_stdio(false);

    char fname[200]; sprintf(fname, "%s.txt", prefix);
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file!" << std::endl;
        return;
    }

    std::vector<u64> datapos;       // starting positions of each sample
    std::vector<DType> dataset;     // positions of 1s
    std::vector<float> datawgt;     // weights of 1s

    std::string line;
    u64 index = 0;                  // Current processing index in `dataset` and `datawgt`
    datapos.push_back(0);           // Starting position of the first sample

    // Read each line (each sparse vector)
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int value;
        DType pos = 0;              // Position within the current vector

        // Parse each value in the sparse vector
        while (iss >> value) {
            if (value == 1) {
                dataset.push_back(pos);    // Store the position of `1`
                datawgt.push_back(1.0f);   // Weight is set to 1.0
                ++index;
            }
            ++pos;
        }
        
        datapos.push_back(index);   // Store the starting position for the next vector
    }
    file.close();

    
    // -------------------------------------------------------------------------
    //  silk: seeding from similar buckets
    // -------------------------------------------------------------------------
    sprintf(fname, "%ssilk_seed.csv", folder);
    if (rank == 0) {
        create_dir(fname);
        
        // fp = fopen(fname, "a+");
        // if (!fp) { printf("Could not open %s\n", fname); exit(1); }
        // fprintf(fp, "K,MSE,MAE,WTime,CTime,");
        // fprintf(fp, "m,h,t,l,k,b,delta,GBeta,LBeta,GAlpha,LAlpha,");
        // fprintf(fp, "#Buckets,#Bins,#Seeds,Phase1,Phase2,Phase3,");
        // fprintf(fp, "EvalWTime,EvalCTime,TotWTime,TotCTime\n");
        // fclose(fp);CHHBF CEzx
    }

    // SILK<DType> *silk = new SILK<DType>(n, d, rank, size, b, gbeta, lbeta, 
    //     galpha, lalpha, prefix, folder, (const DType*) dataset, 
    //     (const float*) datawgt, (const u64*) datapos);
    SILK<DType> *silk = new SILK<DType>(n, d, rank, size, b, gbeta, lbeta, 
        galpha, lalpha, prefix, folder, 
        dataset.data(), datawgt.data(), datapos.data());
    
    int ret = silk->clustering(m, h, t, l, k, delta);
    
    if (rank == 0 && ret == 0) {
        printf("K=%d, MSE=%.4f, MAE=%.4f, SILK=%.2lf Seconds, Eval=%.2lf Seconds\n", 
            g_k, g_mse, g_mae, g_silk_wc_time, g_eval_wc_time);
        printf("\n");
        
        // write the results of each setting to disk
        FILE *fp = fopen(fname, "a+");
        if (!fp) { printf("Could not open %s\n", fname); return; }

        fprintf(fp, "%d,%f,%f,%.4lf,%.4lf,", g_k, g_mse, g_mae, g_silk_wc_time,
            g_silk_cpu_time);
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%g,%g,%g,%g,", m, h, t, l, k, 
            b, delta, gbeta, lbeta, galpha, lalpha);
        fprintf(fp, "%d,%d,%d,%g,%g,%g,", g_tot_buckets, g_tot_bins, g_tot_seeds,
            g_phase1_wc_time, g_phase2_wc_time, g_phase3_wc_time);
        fprintf(fp, "%g,%g,%g,%g\n", g_eval_wc_time, g_eval_cpu_time,
            g_tot_wc_time, g_tot_cpu_time);
        fclose(fp);
    }
    finalize_mpi_comm(mpi_info);
    delete   silk;
    // delete[] datapos;
    // delete[] dataset;
    // delete[] datawgt;
}

// -----------------------------------------------------------------------------
template<class DType>
void kfreqitems_impl(               // k-freqitems implementation
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   k,                            // number of clusters
    int   max_iter,                     // maximum iterations
    int   seeding_algo,                 // seeding algorithm (0-3)
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
    const char *prefix,                 // prefix for data set
    const char *folder)                 // output folder to store output files
{
    MPI_INFO mpi_info;
    init_mpi_comm(mpi_info);
    omp_set_num_threads(THREAD_NUM);
    cudaSetDevice(DEVICE_LOCAL_RANK);
    
    int rank = mpi_info.rank_;
    int size = mpi_info.size_;
    
    // -------------------------------------------------------------------------
    //  read weighted dataset
    // -------------------------------------------------------------------------
    // double start_time = omp_get_wtime();
    // std::ios::sync_with_stdio(false);

    // char fname[200]; sprintf(fname, "%s_%d.bin", prefix, rank);
    // FILE *fp = fopen(fname, "rb");
    // if (!fp) { printf("Rank #%d: cannot open %s\n", rank, fname); exit(1); }

    // // read datapos, dataset, and datawgt
    // u64   *datapos = new u64[n+1]; fread(datapos, sizeof(u64), n+1, fp);
    // u64   N = datapos[n];
    // DType *dataset = new DType[N]; fread(dataset, sizeof(DType), N, fp);
    // float *datawgt = new float[N]; fread(datawgt, sizeof(float), N, fp);
    // fclose(fp);
    
    // double loading_time = omp_get_wtime() - start_time;
    // printf("Rank #%d: n=%d, d=%d, N=%lu, time=%.2lf seconds, path=%s\n\n", 
    //     rank, n, d, N, loading_time, fname);

    // -------------------------------------------------------------------------
    //  read weighted dataset from txt file
    // -------------------------------------------------------------------------
    double start_time = omp_get_wtime();
    std::ios::sync_with_stdio(false);

    char fname[200]; sprintf(fname, "%s.txt", prefix);
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file!" << std::endl;
        return;
    }

    std::vector<u64> datapos;       // starting positions of each sample
    std::vector<DType> dataset;     // positions of 1s
    std::vector<float> datawgt;     // weights of 1s

    std::string line;
    u64 index = 0;                  // Current processing index in `dataset` and `datawgt`
    datapos.push_back(0);           // Starting position of the first sample

    // Read each line (each sparse vector)
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int value;
        DType pos = 0;              // Position within the current vector

        // Parse each value in the sparse vector
        while (iss >> value) {
            if (value == 1) {
                dataset.push_back(pos);    // Store the position of `1`
                datawgt.push_back(1.0f);   // Weight is set to 1.0
                ++index;
            }
            ++pos;
        }
        
        datapos.push_back(index);   // Store the starting position for the next vector
    }
    file.close();
    
    // -------------------------------------------------------------------------
    //  k_freqitems: a new clustering method for sparse weighted data
    // -------------------------------------------------------------------------
    sprintf(fname, "%skfreqitems.csv", folder);
    if (rank == 0) {
        create_dir(fname);
        
        // fp = fopen(fname, "a+");
        // if (!fp) { printf("Could not open %s\n", fname); exit(1); }
        // fprintf(fp, "K,MSE,MAE,WTime,CTime,");
        // fprintf(fp, "k,MaxIter,Iter,GAlpha,LAlpha,Seeding,");
        // fprintf(fp, "InitWTime,InitCTime,IterWTime,IterCTime,TotWTime,TotCTime\n");
        // fclose(fp);
    }
    // KFreqItems<DType> *k_freqitems = new KFreqItems<DType>(n, d, rank, size, 
    //     max_iter, galpha, lalpha, prefix, folder, (const DType*) dataset, 
    //     (const float*) datawgt, (const u64*) datapos);
        KFreqItems<DType> *k_freqitems = new KFreqItems<DType>(n, d, rank, size, 
        max_iter, galpha, lalpha, prefix, folder, 
        dataset.data(), datawgt.data(), datapos.data());
    
    int ret = k_freqitems->clustering(k, seeding_algo, m1, h1, t, m2, h2, b, 
        delta, gbeta, lbeta);
    
    if (rank == 0 && ret == 0) {
        printf("K=%d, MSE=%f, MAE=%f, Time=%.2lf Seconds\n", g_k,  g_mse, g_mae, 
            g_kfreqitems_wc_time);
        printf("Init=%.2lf, Iter=%.2lf, Total=%.2lf Seconds\n", g_init_wc_time,
            g_iter_wc_time, g_tot_wc_time);
        printf("\n");
        
        // write the results of each setting to disk
        FILE *fp = fopen(fname, "a+");
        if (!fp) { printf("Could not open %s\n", fname); return; }

        fprintf(fp, "%d,%f,%f,%.4lf,%.4lf,", g_k, g_mse, g_mae, g_kfreqitems_wc_time, 
            g_kfreqitems_cpu_time);
        fprintf(fp, "%d,%d,%d,%g,%g,%d,", k, max_iter, g_iter, galpha, lalpha, 
            seeding_algo);
        fprintf(fp, "%g,%g,%g,%g,%g,%g\n", g_init_wc_time, g_init_cpu_time, 
            g_iter_wc_time, g_iter_cpu_time, g_tot_wc_time, g_tot_cpu_time);
        fclose(fp);
    }

    finalize_mpi_comm(mpi_info);
    delete   k_freqitems;
    // delete[] datapos;
    // delete[] dataset;
    // delete[] datawgt;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    srand(RANDOM_SEED);             // use a fixed random seed
    
    int   alg    = -1;              // which algorithm (0-1)
    int   n      = -1;              // number of data points
    int   d      = -1;              // data dimension
    int   k      = -1;              // number of seeds
    int   m      = -1;              // maximum iterations
    int   seed   = -1;              // seeding algo (0-3)
    int   m1     = -1;              // #hash tables (1st level minhash)
    int   h1     = -1;              // #concat hash func (1st level minhash)
    int   t      = -1;              // threshold of #point IDs in a bucket
    int   m2     = -1;              // #hash tables (2nd level minhash)
    int   h2     = -1;              // #concat hash func (2nd level minhash)
    int   b      = -1;              // threshold of #bucket IDs in a bin
    int   delta  = -1;              // threshold of #point IDs in a bin
    float gbeta  = -1.0f;           // global \beta
    float lbeta  = -1.0f;           // local  \beta
    float galpha = -1.0f;           // global \alpha
    float lalpha = -1.0f;           // local  \alpha
    char  format[20];               // data format: uint8,uint16,int32,float32
    char  prefix[200];              // prefix for data set
    char  folder[200];              // output folder to store output files
    
    int cnt = 1;
    while (cnt < nargs) {
        if (strcmp(args[cnt], "-alg") == 0) {
            alg = atoi(args[++cnt]); assert(alg >= 0);
        }
        else if (strcmp(args[cnt], "-n") == 0) {
            n = atoi(args[++cnt]); assert(n > 0);
        }
        else if (strcmp(args[cnt], "-d") == 0) {
            d = atoi(args[++cnt]); assert(d > 0);
        }
        else if (strcmp(args[cnt], "-k") == 0) {
            k = atoi(args[++cnt]); assert(k > 0);
        }
        else if (strcmp(args[cnt], "-m") == 0) {
            m = atoi(args[++cnt]); assert(m > 0);
        }
        else if (strcmp(args[cnt], "-s") == 0) {
            seed = atoi(args[++cnt]); assert(seed >= 0);
        }
        else if (strcmp(args[cnt], "-m1") == 0) {
            m1 = atoi(args[++cnt]); assert(m1 > 0);
        }
        else if (strcmp(args[cnt], "-h1") == 0) {
            h1 = atoi(args[++cnt]); assert(h1 > 0);
        }
        else if (strcmp(args[cnt], "-t") == 0) {
            t = atoi(args[++cnt]); assert(t > 0);
        }
        else if (strcmp(args[cnt], "-m2") == 0) {
            m2 = atoi(args[++cnt]); assert(m2 > 0);
        }
        else if (strcmp(args[cnt], "-h2") == 0) {
            h2 = atoi(args[++cnt]); assert(h2 > 0);
        }
        else if (strcmp(args[cnt], "-b") == 0) {
            b = atoi(args[++cnt]); assert(b > 0);
        }
        else if (strcmp(args[cnt], "-D") == 0) {
            delta = atoi(args[++cnt]); assert(delta > 0);
        }
        else if (strcmp(args[cnt], "-GB") == 0) {
            gbeta = atof(args[++cnt]); assert(gbeta >= 0);
        }
        else if (strcmp(args[cnt], "-LB") == 0) {
            lbeta = atof(args[++cnt]); assert(lbeta >= 0);
        }
        else if (strcmp(args[cnt], "-GA") == 0) {
            galpha = atof(args[++cnt]); assert(galpha >= 0);
        }
        else if (strcmp(args[cnt], "-LA") == 0) {
            lalpha = atof(args[++cnt]); assert(lalpha >= 0);
        }
        else if (strcmp(args[cnt], "-F") == 0) {
            strncpy(format, args[++cnt], sizeof(format));
        }
        else if (strcmp(args[cnt], "-P") == 0) {
            strncpy(prefix, args[++cnt], sizeof(prefix));
        }
        else if (strcmp(args[cnt], "-O") == 0) {
            strncpy(folder, args[++cnt], sizeof(folder));
            int len = (int) strlen(folder);
            if (folder[len-1]!='/') { folder[len]='/'; folder[len+1]='\0'; }
        }
        else {
            printf("Parameters error!\n"); usage(); exit(1);
        }
        ++cnt;
    }
    // -------------------------------------------------------------------------
    //  methods 
    // -------------------------------------------------------------------------
    if (alg == 0) {
        if (strcmp(format, "uint8") == 0) {
            silk_impl<u08>(n, d, m1, h1, t, m2, h2, b, delta, gbeta, lbeta, 
                galpha, lalpha, prefix, folder);
        }
        else if (strcmp(format, "uint16") == 0) {
            silk_impl<u16>(n, d, m1, h1, t, m2, h2, b, delta, gbeta, lbeta, 
                galpha, lalpha, prefix, folder);
        }
        else if (strcmp(format, "int32") == 0) {
            silk_impl<int>(n, d, m1, h1, t, m2, h2, b, delta, gbeta, lbeta, 
                galpha, lalpha, prefix, folder);
        }
        else {
            printf("Parameters error!\n"); usage();
        }
    }
    else if (alg == 1) {
        if (strcmp(format, "uint8") == 0) {
            kfreqitems_impl<u08>(n, d, k, m, seed, m1, h1, t, m2, h2, b, delta, 
                gbeta, lbeta, galpha, lalpha, prefix, folder);
        }
        else if (strcmp(format, "uint16") == 0) {
            kfreqitems_impl<u16>(n, d, k, m, seed, m1, h1, t, m2, h2, b, delta, 
                gbeta, lbeta, galpha, lalpha, prefix, folder);
        }
        else if (strcmp(format, "int32") == 0) {
            kfreqitems_impl<int>(n, d, k, m, seed, m1, h1, t, m2, h2, b, delta, 
                gbeta, lbeta, galpha, lalpha, prefix, folder);
        }
        else {
            printf("Parameters error!\n"); usage();
        }
    }
    else {
        printf("Parameters error!\n"); usage();
    }
    return 0;
}
