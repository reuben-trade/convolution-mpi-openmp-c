#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include "conv2d.h"
#include "convolution.h" 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>


/* conv2d with 2D partitioning, halo exchange, and stride */
void conv2d(
    float *f, // input feature map (on rank 0)
    int H, // input height (rows)
    int W, // input width (cols)
    float *g, // kernel
    int kH, int kW, // kernel height/width
    int sH, int sW, // stride height/width
    const char* outfile, // optional output filename
    int argc, char** argv
) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);

    if (W <= 0 || H <= 0 || kW <= 0 || kH <= 0 || sW <= 0 || sH <= 0) {
        if(rank==0) fprintf(stderr,"Invalid dims or stride\n");
        MPI_Finalize();
        return;
    }

    // ---------- 1. Create 2D Cartesian communicator ----------
    int dims[2] = {0,0};
    MPI_Dims_create(num_procs, 2, dims);
    int prows=dims[0], pcols=dims[1];
    int periods[2] = {0,0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    int coords[2]; MPI_Cart_coords(cart_comm, rank, 2, coords);
    int prow = coords[0], pcol = coords[1];

    // ---------- 2. Compute output block per rank ----------
    int padH = kH - 1, padW = kW - 1;
    int out_H = (W + padH - kH) / sH + 1;
    int out_W = (H + padW - kW) / sW + 1;

    // Partition Output space
    int base_rows = out_H / prows; 
    int base_cols = out_W / pcols;    int my_out_rows = base_rows + (prow<rem_rows?1:0);

    int row_offset = (prow<rem_rows)?prow*(base_rows+1):(rem_rows*(base_rows+1)+(prow-rem_rows)*base_rows);

    int base_cols = H / pcols; int rem_cols = H % pcols;
    int my_out_cols = base_cols + (pcol<rem_cols?1:0);
    int col_offset = (pcol<rem_cols)?pcol*(base_cols+1):(rem_cols*(base_cols+1)+(pcol-rem_cols)*base_cols);

    // ---------- 3. Compute needed local input size including halos ----------
    int in_rows_needed = (my_out_rows-1)*sH + kH;
    int in_cols_needed = (my_out_cols-1)*sW + kW;

    // Halo computation (number of extra rows/cols needed from neighbors)
    int halo_top = (prow>0)?kH-1:0;
    int halo_bottom = (prow<prows-1)?kH-1:0;
    int halo_left = (pcol>0)?kW-1:0;
    int halo_right = (pcol<pcols-1)?kW-1:0;

    int local_rows = in_rows_needed + halo_top + halo_bottom;
    int local_cols = in_cols_needed + halo_left + halo_right;

    // Allocate local buffer (with halos)
    float *local_buf = NULL;
    int err = posix_memalign((void**)&local_buf, 64, (size_t)local_rows * local_cols * sizeof(float));
    if (err != 0) {
        fprintf(stderr, "posix_memalign failed with error code %d\n", err);
        exit(1);
    }
    for(size_t i=0;i<(size_t)local_rows*local_cols;++i) local_buf[i]=0.0f;

    // Allocate kernel and broadcast
    float *kernel = malloc((size_t)kH*kW*sizeof(float));
    if(!kernel){ fprintf(stderr,"malloc kernel failed\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    if(rank==0){ memcpy(kernel,g,(size_t)kH*kW*sizeof(float)); free(g); }
    MPI_Bcast(kernel,kH*kW,MPI_FLOAT,0,MPI_COMM_WORLD);

    // ---------- 4. Scatter input blocks to ranks ----------
    // Each rank receives its interior (without halos) from rank 0
    int interior_rows = in_rows_needed;
    int interior_cols = in_cols_needed;
    float *local_interior_ptr = local_buf + halo_top*local_cols + halo_left;

    if(rank==0){
        // Build full padded input
        int padH = kH-1, padW = kW-1;
        int aH = W + padH; int aW = H + padW;
        float *pA = NULL;
        int err = posix_memalign((void**)&pA, 64, (size_t)aH * aW * sizeof(float));
        if (err != 0) {
            fprintf(stderr, "posix_memalign failed with error code %d\n", err);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }        
        for(size_t i=0;i<(size_t)aH*aW;++i) pA[i]=0.0f;
        for(int r=0;r<W;++r) for(int c=0;c<H;++c)
            pA[(r+padH/2)*aW + (c+padW/2)] = f[r*H + c];
        free(f);

        // Send interior blocks to ranks
        for(int rnk=0;rnk<num_procs;++rnk){
            int rcoords[2]; MPI_Cart_coords(cart_comm,rnk,2,rcoords);
            int rprow=rcoords[0], rpcol=rcoords[1];
            int r_out_rows = base_rows + (rprow<rem_rows?1:0);
            int r_out_cols = base_cols + (rpcol<rem_cols?1:0);
            int r_in_rows = (r_out_rows-1)*sH + kH;
            int r_in_cols = (r_out_cols-1)*sW + kW;
            int r_row_offset = (rprow<rem_rows)?rprow*(base_rows+1):(rem_rows*(base_rows+1)+(rprow-rem_rows)*base_rows);
            int r_col_offset = (rpcol<rem_cols)?rpcol*(base_cols+1):(rem_cols*(base_cols+1)+(rpcol-rem_cols)*base_cols);

            // copy from pA into buffer
            if(rnk==0){
                for(int rr=0;rr<r_in_rows;++rr){
                    memcpy(local_interior_ptr + rr*local_cols,
                           pA + (r_row_offset* sH + rr)*aW + r_col_offset*sW, r_in_cols*sizeof(float));
                }
            }else{
                float *tmp = malloc((size_t)r_in_rows*r_in_cols*sizeof(float));
                for(int rr=0;rr<r_in_rows;++rr)
                    memcpy(tmp + rr*r_in_cols,
                           pA + (r_row_offset*sH + rr)*aW + r_col_offset*sW,
                           r_in_cols*sizeof(float));
                MPI_Send(tmp,(int)((size_t)r_in_rows*r_in_cols),MPI_FLOAT,rnk,12345,MPI_COMM_WORLD);
                free(tmp);
            }
        }
        free(pA);
    }else{
        float *tmp = malloc((size_t)interior_rows*interior_cols*sizeof(float));
        MPI_Recv(tmp,(int)((size_t)interior_rows*interior_cols),MPI_FLOAT,0,12345,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(int rr=0;rr<interior_rows;++rr)
            memcpy(local_interior_ptr + rr*local_cols, tmp + rr*interior_cols, interior_cols*sizeof(float));
        free(tmp);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---------- 5. Halo exchange (vertical and horizontal) ----------
    int nbr_up, nbr_down, nbr_left, nbr_right;
    MPI_Cart_shift(cart_comm,0,1,&nbr_up,&nbr_down);
    MPI_Cart_shift(cart_comm,1,1,&nbr_left,&nbr_right);

    MPI_Request reqs[8]; int req_count=0;

    // vertical halo
    if(nbr_up!=MPI_PROC_NULL && halo_top>0)
        MPI_Irecv(local_buf, halo_top*local_cols, MPI_FLOAT, nbr_up, 2001, MPI_COMM_WORLD, &reqs[req_count++]);
    if(nbr_down!=MPI_PROC_NULL && halo_bottom>0)
        MPI_Irecv(local_buf + (halo_top+interior_rows)*local_cols, halo_bottom*local_cols, MPI_FLOAT, nbr_down, 2002, MPI_COMM_WORLD, &reqs[req_count++]);
    if(nbr_up!=MPI_PROC_NULL && halo_top>0)
        MPI_Isend(local_interior_ptr, halo_top*local_cols, MPI_FLOAT, nbr_up, 2002, MPI_COMM_WORLD, &reqs[req_count++]);
    if(nbr_down!=MPI_PROC_NULL && halo_bottom>0)
        MPI_Isend(local_interior_ptr + (interior_rows - halo_bottom)*local_cols, halo_bottom*local_cols, MPI_FLOAT, nbr_down, 2001, MPI_COMM_WORLD, &reqs[req_count++]);

    // horizontal halo
    if(halo_left>0){
        float *sendbuf = malloc(interior_rows*halo_left*sizeof(float));
        float *recvbuf = malloc(interior_rows*halo_left*sizeof(float));
        for(int rr=0;rr<interior_rows;++rr)
            memcpy(sendbuf+rr*halo_left, local_interior_ptr + rr*local_cols, halo_left*sizeof(float));
        if(nbr_left!=MPI_PROC_NULL)
            MPI_Irecv(recvbuf, interior_rows*halo_left, MPI_FLOAT, nbr_left, 3001, MPI_COMM_WORLD, &reqs[req_count++]);
        if(nbr_left!=MPI_PROC_NULL)
            MPI_Isend(sendbuf, interior_rows*halo_left, MPI_FLOAT, nbr_left, 3002, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        // unpack left halo
        if(nbr_left!=MPI_PROC_NULL){
            for(int rr=0;rr<interior_rows;++rr)
                memcpy(local_interior_ptr + rr*local_cols - halo_left, recvbuf + rr*halo_left, halo_left*sizeof(float));
        }
        free(sendbuf); free(recvbuf);
        req_count=0;
    }

    if(halo_right>0){
        float *sendbuf = malloc(interior_rows*halo_right*sizeof(float));
        float *recvbuf = malloc(interior_rows*halo_right*sizeof(float));
        for(int rr=0;rr<interior_rows;++rr)
            memcpy(sendbuf + rr*halo_right, local_interior_ptr + rr*local_cols + (interior_cols - halo_right), halo_right*sizeof(float));
        if(nbr_right!=MPI_PROC_NULL)
            MPI_Irecv(recvbuf, interior_rows*halo_right, MPI_FLOAT, nbr_right, 3002, MPI_COMM_WORLD, &reqs[req_count++]);
        if(nbr_right!=MPI_PROC_NULL)
            MPI_Isend(sendbuf, interior_rows*halo_right, MPI_FLOAT, nbr_right, 3001, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        // unpack right halo
        if(nbr_right!=MPI_PROC_NULL){
            for(int rr=0;rr<interior_rows;++rr)
                memcpy(local_interior_ptr + rr*local_cols + interior_cols, recvbuf + rr*halo_right, halo_right*sizeof(float));
        }
        free(sendbuf); free(recvbuf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---------- 6. Compute local convolution ----------
    float *local_output = malloc((size_t)my_out_rows*my_out_cols*sizeof(float));
    if(!local_output){ fprintf(stderr,"malloc local_output failed\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    #pragma omp parallel for collapse(2) schedule(static)
    for(int r=0;r<my_out_rows;++r){
        for(int c=0;c<my_out_cols;++c){
            float acc=0.0f;
            int in_r = r*sH;
            int in_c = c*sW;
            for(int kr=0;kr<kH;++kr){
                float *in_row_ptr = local_interior_ptr + (in_r+kr)*local_cols + in_c;
                float *k_row_ptr = kernel + kr*kW;
                #pragma omp simd reduction(+:acc)
                for(int kc=0;kc<kW;++kc){
                    acc += in_row_ptr[kc] * k_row_ptr[kc];
                }
            }
            local_output[r*my_out_cols + c] = acc;
        }
    }

    // ---------- 7. Gather outputs to rank 0 ----------
    if(rank==0){
        float *full_output = malloc((size_t)out_H*out_W*sizeof(float));
        for(int rr=0;rr<my_out_rows;++rr)
            memcpy(full_output + (row_offset+rr)*H + col_offset, local_output + rr*my_out_cols, my_out_cols*sizeof(float));
        for(int rnk=1;rnk<num_procs;++rnk){
            int rcoords[2]; MPI_Cart_coords(cart_comm,rnk,2,rcoords);
            int rprow=rcoords[0], rpcol=rcoords[1];
            int r_out_rows = base_rows + (rprow<rem_rows?1:0);
            int r_out_cols = base_cols + (rpcol<rem_cols?1:0);
            float *tmp = malloc((size_t)r_out_rows*r_out_cols*sizeof(float));
            MPI_Recv(tmp, r_out_rows*r_out_cols, MPI_FLOAT, rnk, 5555, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int r_row_offset = (rprow<rem_rows)?rprow*(base_rows+1):(rem_rows*(base_rows+1)+(rprow-rem_rows)*base_rows);
            int r_col_offset = (rpcol<rem_cols)?rpcol*(base_cols+1):(rem_cols*(base_cols+1)+(rpcol-rem_cols)*base_cols);
            for(int rr=0;rr<r_out_rows;++rr)
                memcpy(full_output + (r_row_offset+rr)*H + r_col_offset, tmp + rr*r_out_cols, r_out_cols*sizeof(float));
            free(tmp);
        }
        if(outfile) write_matrix_to_file_text(outfile, full_output, out_W, out_H);
        free(full_output);
    }else{
        MPI_Send(local_output, my_out_rows*my_out_cols, MPI_FLOAT, 0, 5555, MPI_COMM_WORLD);
    }

    // cleanup
    free(local_buf); free(local_output); free(kernel);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
}


void _conv2d(
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

    printf("CONV BABY!!\n");
    // Padding Dims
    int pW = kW - 1;        
    int pH = kH - 1;

    // Padded Matrix Dims
    int aW = W+pW;        
    int aH = H+pH;

    // Get amount of threads available for parallelisation
    int n_threads = omp_get_max_threads(); 
    // int n_threads = 1;

    /* Allocating memory within each processor for the padded matrix and kernel */
    float* pA = malloc(aW*aH*sizeof(float)); 
    memset(pA, 0, aW*aH*sizeof(float)); // init padded matrix to 0s
    float* kernel = malloc(kW*kH*sizeof(float));
    float* local_output = NULL;

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