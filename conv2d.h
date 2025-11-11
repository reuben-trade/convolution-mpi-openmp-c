#ifndef CONV2D_H
#define CONV2D_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// Main convolution function
void conv2d(
    float *f,
    int H, int W,
    float *g,
    int kH, int kW,
    int sH, int sW,
    const char* outfile,
    int argc,
    char **argv
);

// Include other module headers
#include "matrix_ops.h"
#include "file_io.h"
#include "utils.h"

#endif // CONV2D_H