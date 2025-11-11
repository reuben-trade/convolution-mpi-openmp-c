#include "conv2d.h"

int main(int argc, char **argv) {

    // Default values
    int W = 100, H = 100, kW = 30, kH = 30;
    int sW = 1, sH = 1;         // <--- default stride
    int is_random = 1;

    const char *infile = NULL, *kinfile = NULL, *outfile = NULL;
    float *f1d = NULL, *g1d = NULL;

    // Parse command-line arguments including stride
    parse_command_line(argc, argv, &W, &H, &kW, &kH, &sW, &sH, &is_random, &infile, &kinfile, &outfile);

    // Input matrix
    if (infile) {
        int rW, rH;
        read_matrix_from_file_text(infile, &rW, &rH, f1d);
        if (!f1d) { fprintf(stderr, "Failed to read matrix from %s\n", infile); return 1; }
        W = rW; H = rH;
    } else {
        f1d = malloc(sizeof(float)*W*H);
        if (!f1d) { fprintf(stderr, "Failed to allocate input matrix\n"); return 1; }
        generate_matrix_1D_memory(W, H, is_random, f1d);
    }

    // Kernel matrix
    if (kinfile) {
        int rkW, rkH;
        read_matrix_from_file_text(kinfile, &rkW, &rkH, g1d);
        if (!g1d) { fprintf(stderr, "Failed to read kernel from %s\n", kinfile); free(f1d); return 1; }
        kW = rkW; kH = rkH;
    } else {
        g1d = malloc(sizeof(float)*kW*kH);
        if (!g1d) { fprintf(stderr, "Failed to allocate kernel matrix\n"); free(f1d); return 1; }
        generate_matrix_1D_memory(kW, kH, is_random, g1d);
    }

    // Trigger convolution (pass stride)
    conv2d(f1d, H, W, g1d, kH, kW, sH, sW, outfile, argc, argv);

    return 0;
}
