#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Command line parsing
void parse_command_line(int argc, char **argv,
                       int *W, int *H, int *kW, int *kH, int *sW, int *sH, int *is_random,
                       const char **infile, const char **kinfile, const char **outfile);

#endif // UTILS_H