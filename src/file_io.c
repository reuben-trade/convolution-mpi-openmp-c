#include "file_io.h"

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

void write_matrix_to_file_text(const char *filename, const float *matrix, int W, int H) {
    /* Method to write a matrix to an output file */
    if (!filename || !matrix || W <= 0 || H <= 0) return;
    FILE *f = fopen(filename, "w");
    if (!f) return;
    if (fprintf(f, "%d %d\n", W, H) < 0) {
        fclose(f);
    }
    long total = (long)W * (long)H;
    for (long i = 0; i < total; ++i) {
        if (fprintf(f, "%.9g%c", matrix[i], (i % H == H-1) ? '\n' : ' ') < 0) {
            fclose(f);
            return;
        }
    }
    fclose(f);
    return;
}

void read_matrix_from_file_text(const char *filename, int *outW, int *outH, float* matrix) {
    /* Method to read input files of matricies and kernels */
    if (!filename || !outW || !outH) {
        printf("DEBUG: NULL parameter passed\n");
        return;
    }

    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("DEBUG: Failed to open file: %s\n", filename);
        perror("fopen");
        return;
    }

    int W, H;
    int scan_result = fscanf(f, "%d %d", &W, &H);

    matrix = malloc(W*H*sizeof(float));
    
    if (scan_result != 2) {
        printf("DEBUG: Failed to read dimensions (expected 2 values, got %d)\n", scan_result);
        fclose(f);
        return;
    }

    if (W <= 0 || H <= 0) { 
        printf("DEBUG: Invalid dimensions W=%d, H=%d\n", W, H);
        fclose(f); 
        return; 
    }

    long total = (long)W * (long)H;

    // Read all float values sequentially
    for (long i = 0; i < total; ++i) {
        if (fscanf(f, "%f", &matrix[i]) != 1) {
            printf("DEBUG: Failed to read float at position %ld\n", i);
            fclose(f);
            return;
        }
    }

    fclose(f);
    *outW = W;
    *outH = H;
}
