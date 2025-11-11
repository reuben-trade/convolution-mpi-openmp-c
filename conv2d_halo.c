#include "conv2d.h"
#include "convolution.h" 

void conv2d(
    float *f, // input feature map
    int H, // input height,
    int W, // input width
    float *g, // input kernel
    int kH, // kernel height
    int kW, // kernel width
    const char* outfile, // file to write to 
    int argc, /* argc & argv from main to init MPI */
    char** argv

) { 

    int rank, num_procs; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Padded Matrix Dims
    int aW = W+pW;        
    int aH = H+pH;

    // Get amount of threads available for parallelisation
    int n_threads = omp_get_max_threads(); 
    // int n_threads = 1;

    /* Create Grid Dimensions */
    int dims[2] = {0, 0}; // let MPI choose the 2D composition
    MPI_Dims_create(num_procs, 2, dims); // create grid dimensions 

    /* Set up Grid parameters */
    int prows = dims[0], pcols = dims[1] // # process rows & cols 
    int periods[2] = {0, 0}; // non-periodic/wrap-around - edges don't connect

    /* Create Cartesian Communicator */
    MPI_Comm cart_comm; // define communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm); // let MPI order ranks 

    /* Turn MPI Rank into 2D grid position */
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords;)
    int prow = coords[0], pcol = coords[1]; // each process knows its prow & pcol position in grid

    /* Find each processor's grid dimensions */
    int base_rows = H / prows; 
    int rem_rows = H % prows; // remainder
    int p_rows = base_rows + (prow < rem_rows ? 1: 0); // individual processor's rows 
    int row_offset = (prow < rem_rows) ? // If this process gets extra row 
                    (prow * (base_rows+1)) : // they have base rows + 1 row
                    (rem_rows * (base_rows+1) + (prow - rem_rows) * base_rows); // else: the first rem_rows have 1 + base_rows, the remainder (until this p) have just base_rows
   
    
    // cols
    int base_cols = W / pcols; // How many cols each p is guaraneed
    int rem_cols = W % pcols; // # ps with a remainder col
    int p_cols = base_cols + (pcol < rem_rows ? 1: 0); // proc's #cols
    int col_offset = (pcol < rem_cols) ?  // col ofset -> where p starts in grid
                        (base_cols+1)*pcol:
                        (rem_cols*(base_cols+1) + (pcols-rem_cols)*base_cols);


    /* halo widths: we use full (kH-1) and (kW-1) on both sides */
    int halo_top = kH - 1;
    int halo_bottom = kH - 1;
    int halo_left = kW - 1;
    int halo_right = kW - 1;

    float* kernel = malloc(kW*kH*sizeof(float));

    /* allocate kernel on all ranks and broadcast from root */
    float *kernel = malloc((size_t)kW * kH * sizeof(float));
    if (!kernel) { fprintf(stderr,"malloc kernel failed\n"); MPI_Abort(MPI_COMM_WORLD,1); }

    if (rank == 0) {
        memcpy(kernel, g, (size_t)kW * kH * sizeof(float));
        free(g);
    }
    MPI_Bcast(kernel, kW*kH, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Padding Dims
    int pW = kW - 1;        
    int pH = kH - 1;

    int recv_rows = p_rows;
    int recv_cols = p_cols;

    
    /* Declaring variables for each processor to store the size of their 'chunk' of the output array and the index at which they start */
    int local_chunk_size, local_start_idx;

    /* Initialise arrays to to NULL for all ranks, only populate in rank 0 */
    int *per_proc_work = NULL;        
    int *proc_start_index = NULL;

    /* Processor with rank 0 set up */
    if (rank == 0) {

        // ###TODO: determine if it's necessary to pad the matrix this way 
        /* Pad the matrix */
        apply_padding(f, W, H, pW, pH, pA);
        free(f); 

        /* Copy g into kernel variable for later broadcasting  */
        memcpy(kernel, g, sizeof(float)*kW*kH);
        printf("copied memory");
        free(g);

        printf("Padded Matrix Dims: %dx%d\n", aW, aH);

        /* Initialise arrays to store the number of output elements each processor will fill & its starting index inside the output matrix */
        per_proc_work = malloc(num_procs * sizeof(int));        
        proc_start_index = malloc(num_procs * sizeof(int));

        long long total_cells = (long long)W * (long long)H; // total output elements
        
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

    /* Store local output */
    local_output = malloc(local_chunk_size*sizeof(float));

    /* Each rank performs convolution on their subsect of data and stores in local output array */
    float comp_time = perform_convulution_chunk(n_threads, local_start_idx, local_chunk_size, H, kW, kH, pH, pA, kernel, local_output);
    printf("Processor %d: %f\n", rank, comp_time);

    const char* filename = "output.bin";
    /* Fast, binary collective write */
    double io_time = collective_io_write(filename, local_start_idx, local_chunk_size, local_output, MPI_COMM_WORLD);
    printf("[IO]Processor %d: %f\n", rank, io_time);
    
    /* Determine if user wants to write to output file in human-readable-format */
    if (outfile) {
        /* Allocation for the full output array is only needed for the rank 0 */
        float* full_output = NULL;
        if (rank == 0) { full_output = malloc(W*H*sizeof(float)); }

        /* Rank 0 gathers all output buffers in order */ 
        MPI_Gatherv(local_output, local_chunk_size, MPI_FLOAT, full_output, per_proc_work, proc_start_index, MPI_FLOAT, 0, MPI_COMM_WORLD); // Gatherv: all elements can send variable amounts of data
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) { 
            /* Slower, text collective write */
            write_matrix_to_file_text(outfile, full_output, W, H); 
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
    MPI_Finalize();
    
    
}