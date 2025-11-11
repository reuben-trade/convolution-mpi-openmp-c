#include "utils.h"

void parse_command_line(int argc, char **argv,
                        int *W, int *H, int *kW, int *kH, int *sW, int *sH, int *is_random,
                        const char **infile, const char **kinfile, const char **outfile) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-W width] [-H height] [-kW kw] [-kH kh] [-sW strideW] [-sH strideH] [-r] [-i infile] [-ki kinfile] [-o outfile]\n", argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "-W") == 0 && i+1 < argc) {
            *W = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-H") == 0 && i+1 < argc) {
            *H = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-kW") == 0 && i+1 < argc) {
            *kW = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-kH") == 0 && i+1 < argc) {
            *kH = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-sW") == 0 && i+1 < argc) {
            *sW = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-sH") == 0 && i+1 < argc) {
            *sH = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            *is_random = 1;
        } else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) {
            *infile = argv[++i];
        } else if (strcmp(argv[i], "-ki") == 0 && i+1 < argc) {
            *kinfile = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
            *outfile = argv[++i];
        } else {
            fprintf(stderr, "Unknown or malformed argument: %s\n", argv[i]);
            fprintf(stderr, "Use --help for usage.\n");
            exit(1);
        }
    }
}
