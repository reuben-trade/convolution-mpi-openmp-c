#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <limits.h>


float perform_convulution_chunk(int num_threads, int start_idx, int chunk_size, int H, int kW, int kH, int pH, float* padded_matrix, float* kernel, float* output);

#endif // PERFORMANCE_H