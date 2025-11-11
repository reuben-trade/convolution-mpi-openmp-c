#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

// Matrix generation functions
void generate_matrix_1D_memory(int W, int H, int is_random, float* output);

// Matrix manipulation functions
void apply_padding(const float *A, int aW, int aH, int pW, int pH, float* pA);

#endif // MATRIX_OPS_H