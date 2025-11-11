#ifndef FILE_IO_H
#define FILE_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// File I/O functions
double collective_io_write(const char* filename, int start_idx,  int chunk_size, const float *local_out, MPI_Comm comm);
void write_matrix_to_file_text(const char *filename, const float *matrix, int W, int H);
void read_matrix_from_file_text(const char *filename, int *outW, int *outH, float* matrix);

#endif // FILE_IO_H