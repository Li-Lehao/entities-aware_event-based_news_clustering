#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <vector>
#include <unordered_map>

#include "def.h"
#include "util.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
//  Broder, Andrei Z., Moses Charikar, Alan M. Frieze, and Michael Mitzenmacher.
//  "Min-wise independent permutations." In Proceedings of the thirtieth annual 
//  ACM Symposium on Theory of Computing (STOC), pp. 327-336. 1998.
// -----------------------------------------------------------------------------
template<class DType>
void minhash(                       // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const DType *dataset,               // data/bucket set
    const u64   *datapos,               // data/bucket position
    int   *hash_results);               // hash results (return)

// -----------------------------------------------------------------------------
//  Ioffe, Sergey. "Improved consistent sampling, weighted minhash and l1 
//  sketching." In 2010 IEEE international conference on data mining, pp. 
//  246-255. IEEE, 2010.
// -----------------------------------------------------------------------------
template<class DType>
void weighted_minhash(              // weighted minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data
    int   d,                            // data dimension
    int   prime,                        // prime number for data
    int   m,                            // #hash tables
    int   l,                            // pair wise numbers
    int   k,                            // pair wise concatenation
    float *r_k,                         // hash param r_k ~ Gamma(2,1)
    float *c_k,                         // hash param c_k ~ Gamma(2,1)
    float *b_k,                         // hash param b_k ~ Uniform(0,1)
    const DType *dataset,               // data set
    const float *datawgt,               // data weight
    const u64   *datapos,               // data position
    int   *hash_results);               // hash results (return)

} // end namespace clustering
