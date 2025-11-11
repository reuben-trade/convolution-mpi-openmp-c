#include "convolution.h"

float perform_convulution_chunk(int num_threads, int start_idx, int chunk_size, int H, int kW, int kH, int pH, float* padded_matrix, float* kernel, float* output) {
    /* Assesing Convultion Without the Overhead of Storing Metrics*/

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();

    int padded_stride = H+pH;

    #pragma omp parallel
    {

        #pragma omp for schedule(static)
        for (int i=start_idx; i<start_idx+chunk_size; i++) {
                            
                float score = 0.0; 
                int current_row = i/H;
                int current_col = i - current_row*H;

                for (int kw=0; kw<kW; kw++) {

                    // pre-computing base indicies
                    int padded_rowbase = (current_row+kw)*(padded_stride)+current_col; 
                    int kernel_rowbase = kw*kH;

                    #pragma omp simd reduction(+:score)
                    for (int kh=0; kh<kH; kh++) {
                        score += (padded_matrix[padded_rowbase+kh] * kernel[kernel_rowbase+kh]); //2 FLOPS (+&*)
                    }
                }

                // So indexing is 0 based
                output[i-start_idx] = score;        
            }
    }
    double end = omp_get_wtime();
    return end-start;

}
