#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <mpi.h>

void apply_padding(
    const float *A,   // input matrix (row-major, size W x H)
    int W, int H,   // input dimensions: rows (aW) and cols (aH)
    int pW, int pH,   // total padding to add to rows and cols  
    float *pA        // output matrix
) {
    /* Using multiple threads to apply padding of {0} to pW rows & pH cols of the matrix A in parallel */

    if (W <= 0 || H <= 0 || pW < 0 || pH < 0) return;

    int newW = W + pW; // new number of rows
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

double collective_io_write(const char* filename, 
    int start_idx,  
    int chunk_size,
    const float *local_out,  // pointer to the actual output from each processor
    MPI_Comm comm) // collective communicator
{
    double io_start = MPI_Wtime();

    /* Define MPI file handle & file info */
    MPI_File fh;
    MPI_Info info; // pass system level info to MPI file handler
    MPI_Info_create(&info);

    /* Collectively create / open (in write mode) the file across the communicator */
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, info, &fh);
    MPI_Info_free(&info); // no longer needed

    /* Each process calculates their byte offset form start of file */
    MPI_Offset offset = (MPI_Offset)start_idx * sizeof(float);
    /* Each process writes their local buffer (in) starting at their offset */
    MPI_Status st;
    MPI_File_write_at_all(fh, offset, (void*)local_out, (MPI_Count)chunk_size, MPI_FLOAT, &st);
    MPI_File_close(&fh); // close the file across all ranks 
 
    /* Imply barrier so processors wait here */
    MPI_Barrier(comm);
    double io_end = MPI_Wtime();
    return io_end - io_start;

}

int write_matrix_to_file_text(const char *filename, const float *matrix, int W, int H) {
    /* Method to write a matrix to an output file */
    if (!filename || !matrix || W <= 0 || H <= 0) return -1;
    FILE *f = fopen(filename, "w");
    if (!f) return -2;
    if (fprintf(f, "%d %d\n", W, H) < 0) {
        fclose(f);
        return -3;
    }
    long total = (long)W * (long)H;
    for (long i = 0; i < total; ++i) {
        if (fprintf(f, "%.9g%c", matrix[i], (i % H == H-1) ? '\n' : ' ') < 0) {
            fclose(f);
            return -4;
        }
    }
    fclose(f);
    return 0;
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int rank, num_procs; 

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Kernel Dims
    int kW = 3;
    int kH = 3;
    // Matrix Dims
    int W = 1000;
    int H = 1000;

    // Do we write to output file
    int is_output = 0;

    // Padding Dims
    int pW = kW - 1;        
    int pH = kH - 1;

    // Padded Matrix Dims
    int aW = W+pW;        
    int aH = H+pH;

    // Set the number of threads for each processor
    int n_threads = omp_get_max_threads(); 
    // int n_threads = 1;

    /* Allocating memory within each processor for the padded matrix and kernel */
    float* pA = malloc(aW*aH*sizeof(float)); 
    memset(pA, 0, aW*aH*sizeof(float)); // init padded matrix to 0s

    float* kernel = malloc(kW*kH*sizeof(float));  

    /* Declaring variables for each processor to store the size of their 'chunk' of the output array and the index at which they start */
    int local_chunk_size, local_start_idx;

    /* Initialise arrays to to NULL for all ranks, only populate in rank 0 */
    int *per_proc_work = NULL;        
    int *proc_start_index = NULL;

    /* Processor with rank 0 set up */
    if (rank == 0) {

        /* populate the kernel and input matrix with random values */
        generate_matrix_1D_memory(kW, kH, 1, kernel);

        float* A = malloc(W*H*sizeof(float)); // initialise memory - other ranks don't need the input matrix
        generate_matrix_1D_memory(W, H, 1, A);

        // ###TODO: determine if it's necessary to pad the matrix this way 
        /* Pad the matrix */
        apply_padding(A, W, H, pW, pH, pA);
        free(A); 

        printf("Padded Matrix Dims: %dx%d\n", aW, aH);

        /* Initialise arrays to store the number of output elements each processor will fill & its starting index inside the output matrix */
        per_proc_work = malloc(num_procs * sizeof(int));        
        proc_start_index = malloc(num_procs * sizeof(int));

        long long total_cells = (long long)W * (long long)H; // total output elements
        int kernel_size = kW * kH; 

        
        /* How much 'guaranteed' output cells each processor will operate on */
        int even_work_per_proc = (int)( total_cells /  (long long)num_procs );        
        long long remaining_work = total_cells % (long long)num_procs; // leftover work per proc

        // how many EXTRA cells exist in that remaining_work
        int n_extra_cells = (int)remaining_work; // may be 0 ### is this necessary?

        /* Calculate the sum and per processor work for each rank */
        long long sum = 0; // maintain a sum to track where in the array each proc starts
        for (int p = 0; p < num_procs; ++p) {

            long long cells_for_p = (long long)even_work_per_proc + (p < n_extra_cells ? 1 : 0);
            per_proc_work[p] = (int)cells_for_p;

            proc_start_index[p] = sum;
            sum += per_proc_work[p];
        }

    }

    // ## TODO: subdivide communicators, you can send necessary data to necessary communicators for large arrays
    /* Share the necessary data to all processes */ 
    MPI_Scatter(per_proc_work, 1, MPI_INT, &local_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD); // # elements each proc will work on
    MPI_Scatter(proc_start_index, 1, MPI_INT, &local_start_idx, 1, MPI_INT, 0, MPI_COMM_WORLD); // index proc begins operating on

    MPI_Bcast(pA, aW*aH, MPI_FLOAT, 0, MPI_COMM_WORLD); // full padded matrix
    MPI_Bcast(kernel, kW*kH, MPI_FLOAT, 0, MPI_COMM_WORLD); // full kernel 

    /* Each rank performs convolution on their subsect of data and stores in local output array */
    float* output_array = malloc(local_chunk_size*sizeof(float));
    float comp_time = perform_convulution_chunk(n_threads, local_start_idx, local_chunk_size, H, kW, kH, pH, pA, kernel, output_array);
    printf("Processor %d: %f\n", rank, comp_time);

    const char* filename = "output.bin";
    /* Fast, binary collective write */
    double io_time = collective_io_write(filename, local_start_idx, local_chunk_size, output_array, MPI_COMM_WORLD);
    printf("[IO]Processor %d: %f\n", rank, io_time);
    
    /* Determine if user wants to write to output file in human-readable-format */
    if (is_output) {

        /* Allocation for the full output array is only needed for the rank 0 */
        float* full_output = NULL;
        if (rank == 0) { full_output = malloc(W*H*sizeof(float)); }

        /* Rank 0 gathers all output buffers in order */ 
        MPI_Gatherv(output_array, local_chunk_size, MPI_FLOAT, full_output, per_proc_work, proc_start_index, MPI_FLOAT, 0, MPI_COMM_WORLD); // Gatherv: all elements can send variable amounts of data
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) { 
            /* Slower, text collective write */
            write_matrix_to_file_text("output.txt", full_output, W, H); 
        }

        free(full_output);

    }

     if (rank==0) {
        /* Free arrays after Scatter */
        free(per_proc_work);
        free(proc_start_index);
    }

    free(pA);
    free(kernel);
    free(output_array);
    MPI_Finalize();

}   