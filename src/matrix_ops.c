#include "matrix_ops.h"

void apply_padding(
    const float *A,   // input matrix (row-major, size W x H)
    int W, int H,   // input dimensions: rows (aW) and cols (aH)
    int pW, int pH,   // total padding to add to rows and cols  
    float *pA        // output matrix
) {
    /* Using multiple threads to apply padding of {0} to pW rows & pH cols of the matrix A in parallel */

    if (W <= 0 || H <= 0 || pW < 0 || pH < 0) return;

    int newH = H + pH; // new number of columns

    int p_top = pW/2; // # padded rows before first original row
    int p_left = pH/2; // # padded rows before first original col 


    // Defining a parellised region 
    #pragma omp parallel   
    {    
        #pragma omp for collapse(2) schedule(static)
        for (int i=0; i<W; ++i) {
            for (int j=0; j<H; ++j) {
                pA[(i+p_top)*newH+j+p_left] = A[i*H+j]; 
            }
        
        }            
    }
}


void generate_matrix_1D_memory(
    int W, // # Rows
    int H, // # Cols
    int is_random, // 0: array of 0.6f, 1: array of random values: [0, 1]
    float* output // output array to save to 
) {

    // fill with values float32 values
    for (int i=0; i<W*H; i++) {
        if (is_random) {
            output[i] = (float)rand()/RAND_MAX; 
        }
        else {
            output[i] = 0.0;            
        }
    }
}