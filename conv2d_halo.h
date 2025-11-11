#ifndef CONV2D_HALO_H
#define CONV2D_HALO_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// Main convolution function
void conv2d(
    float *f,    // input feature map
    int H,        // input height
    int W,        // input width
    float *g,    // input kernel
    int kH,       // kernel height
    int kW,       // kernel width
    const char* outfile,
    int argc,
    char** argv
);

// Include other module headers
#include "matrix_ops.h"
#include "file_io.h"
#include "utils.h"

#endif // CONV2D_HALO_H