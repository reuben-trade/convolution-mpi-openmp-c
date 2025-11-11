
/*
 * assign_tiles_even_spacing_balanced.c
 *
 * Improved tile assignment: searches possible (cols x rows) grids and selects
 * the one that yields the most even per-rank compute-area after applying the
 * "make non-last columns/rows multiples of kW/kH, last absorbs leftover" rule.
 *
 * Returns: ProcTile * array of length num_procs (malloc'd) or NULL on failure.
 *
 * Keeps the same ProcTile layout as before.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

typedef struct {
    int rank;
    int col, row;
    int start_x, start_y;
    int tile_w, tile_h;
    int dW, dH;
    int read_x, read_y;
    int read_w, read_h;
} ProcTile;

/* helper: produce tiles for a given cols x rows config (very similar to earlier function)
 * but writes to a provided tiles_out array of size num_procs.
 * returns 0 on success, non-zero on allocation failure.
 */
static int build_tiles_for_grid(int padded_w, int padded_h, int kW, int kH,
                                int cols, int rows, int num_procs, ProcTile *tiles_out) {
    if (cols <= 0 || rows <= 0) return -1;

    int rad_x = (kW > 0) ? (kW / 2) : 0;
    int rad_y = (kH > 0) ? (kH / 2) : 0;

    // initial even split (distribute +1 across first rem bins)
    int base_col = (cols > 0) ? (padded_w / cols) : padded_w;
    int rem_col  = (cols > 0) ? (padded_w % cols) : 0;
    int base_row = (rows > 0) ? (padded_h / rows) : padded_h;
    int rem_row  = (rows > 0) ? (padded_h % rows) : 0;

    int *col_w = (int*)calloc((size_t)cols, sizeof(int));
    int *row_h = (int*)calloc((size_t)rows, sizeof(int));
    if (!col_w || !row_h) { free(col_w); free(row_h); return -1; }

    for (int c = 0; c < cols; ++c) col_w[c] = base_col + (c < rem_col ? 1 : 0);
    for (int r = 0; r < rows; ++r) row_h[r] = base_row + (r < rem_row ? 1 : 0);

    // Floor non-last columns to multiples of kW; last gets leftover
    if (kW > 1) {
        int sum_cols = 0;
        for (int c = 0; c < cols - 1; ++c) {
            col_w[c] = (col_w[c] / kW) * kW;
            if (col_w[c] < 0) col_w[c] = 0;
            sum_cols += col_w[c];
        }
        col_w[cols - 1] = padded_w - sum_cols;
        if (col_w[cols - 1] < 0) col_w[cols - 1] = 0;
    }

    // Floor non-last rows to multiples of kH; last gets leftover
    if (kH > 1) {
        int sum_rows = 0;
        for (int r = 0; r < rows - 1; ++r) {
            row_h[r] = (row_h[r] / kH) * kH;
            if (row_h[r] < 0) row_h[r] = 0;
            sum_rows += row_h[r];
        }
        row_h[rows - 1] = padded_h - sum_rows;
        if (row_h[rows - 1] < 0) row_h[rows - 1] = 0;
    }

    // prefix sums for starts
    int *col_x = (int*)calloc((size_t)cols, sizeof(int));
    int *row_y = (int*)calloc((size_t)rows, sizeof(int));
    if (!col_x || !row_y) { free(col_w); free(row_h); free(col_x); free(row_y); return -1; }
    int acc = 0;
    for (int c = 0; c < cols; ++c) { col_x[c] = acc; acc += col_w[c]; }
    acc = 0;
    for (int r = 0; r < rows; ++r) { row_y[r] = acc; acc += row_h[r]; }

    // populate tiles_out for ranks [0..num_procs-1]
    for (int rank = 0; rank < num_procs; ++rank) {
        ProcTile t;
        t.rank = rank;
        t.row = rank / cols;
        t.col = rank % cols;

        if (t.row >= rows || t.col >= cols) {
            // out of grid: zero-size tile
            t.start_x = t.start_y = 0;
            t.tile_w = t.tile_h = 0;
            t.dW = t.dH = 0;
            t.read_x = t.read_y = t.read_w = t.read_h = 0;
            tiles_out[rank] = t;
            continue;
        }

        t.start_x = col_x[t.col];
        t.start_y = row_y[t.row];
        t.tile_w = col_w[t.col];
        t.tile_h = row_h[t.row];
        t.dW = (kW > 0) ? (t.tile_w % kW) : 0;
        t.dH = (kH > 0) ? (t.tile_h % kH) : 0;

        t.read_x = t.start_x - rad_x; if (t.read_x < 0) t.read_x = 0;
        t.read_y = t.start_y - rad_y; if (t.read_y < 0) t.read_y = 0;
        int rx_end = t.start_x + t.tile_w + rad_x; if (rx_end > padded_w) rx_end = padded_w;
        int ry_end = t.start_y + t.tile_h + rad_y; if (ry_end > padded_h) ry_end = padded_h;
        t.read_w = (rx_end > t.read_x) ? (rx_end - t.read_x) : 0;
        t.read_h = (ry_end > t.read_y) ? (ry_end - t.read_y) : 0;

        tiles_out[rank] = t;
    }

    free(col_w); free(row_h); free(col_x); free(row_y);
    return 0;
}

/* compute metric of imbalance for the current tiles: we measure max_area - min_area
 * for the first num_procs actual tiles (row-major). This is simple and directly
 * reflects how uneven the per-rank compute load is (area = tile_w * tile_h).
 */
static long evaluate_tiles_imbalance(const ProcTile *tiles, int num_procs) {
    long maxA = LONG_MIN;
    long minA = LONG_MAX;
    for (int r = 0; r < num_procs; ++r) {
        const ProcTile *t = &tiles[r];
        long area = (long)t->tile_w * (long)t->tile_h;
        if (area > maxA) maxA = area;
        if (area < minA) minA = area;
    }
    // If all tiles are zero (degenerate), return large value to avoid choosing this grid.
    if (maxA == LONG_MIN || minA == LONG_MAX) return LONG_MAX;
    return (maxA - minA);
}

/* Main improved function:
 * - Tries all candidate cols = 1..num_procs (rows = ceil(num_procs/cols))
 * - Builds tiles for each candidate using the same multiple-flooring rule
 * - Chooses the candidate with min (max_area - min_area)
 */
ProcTile *assign_tiles_even_spacing_balanced(int padded_w, int padded_h, int kW, int kH, int num_procs) {
    if (num_procs < 1) return NULL;

    ProcTile *best_tiles = (ProcTile*)malloc((size_t)num_procs * sizeof(ProcTile));
    ProcTile *candidate_tiles = (ProcTile*)malloc((size_t)num_procs * sizeof(ProcTile));
    if (!best_tiles || !candidate_tiles) { free(best_tiles); free(candidate_tiles); return NULL; }

    long best_metric = LONG_MAX;
    int best_cols = 1, best_rows = num_procs;

    // Try all possible columns counts
    for (int cols = 1; cols <= num_procs; ++cols) {
        int rows = (num_procs + cols - 1) / cols; // ceil division
        // don't try hopeless grids where cols > padded_w or rows > padded_h?
        // We will still allow them — build_tiles will produce zero-size tiles if needed.
        if (build_tiles_for_grid(padded_w, padded_h, kW, kH, cols, rows, num_procs, candidate_tiles) != 0) {
            continue; // allocation problem for this candidate
        }
        long metric = evaluate_tiles_imbalance(candidate_tiles, num_procs);
        // If metric equal, prefer layout with smaller perimeter (heuristic) to keep "compact" grids:
        if (metric < best_metric) {
            best_metric = metric;
            best_cols = cols;
            best_rows = rows;
            memcpy(best_tiles, candidate_tiles, (size_t)num_procs * sizeof(ProcTile));
            // short-circuit if perfect (zero imbalance)
            if (best_metric == 0) break;
        }
    }

    free(candidate_tiles);
    // If we never set best_tiles (shouldn't happen), fallback to cols=1
    if (best_metric == LONG_MAX) {
        free(best_tiles);
        return NULL;
    }

    // best_tiles holds the selected tiling
    return best_tiles;
}


/* Flatten read region into a contiguous buffer (row-major).
 * image_ptr points to padded image base (row-major ints), padded_stride is padded_w.
 * elem_size is bytes per element (e.g. sizeof(int)).
 * Returns malloc'd buffer or NULL on failure.
 */
void *flatten_read_region(const void *image_ptr, size_t padded_stride, const ProcTile *tile, size_t elem_size) {
    if (!image_ptr || !tile) return NULL;
    if (tile->read_w <= 0 || tile->read_h <= 0) return NULL;

    size_t total = (size_t)tile->read_w * (size_t)tile->read_h;
    void *buf = malloc(total * elem_size);
    if (!buf) return NULL;

    const unsigned char *src = (const unsigned char*)image_ptr;
    unsigned char *dst = (unsigned char*)buf;
    for (int r = 0; r < tile->read_h; ++r) {
        const void *row_src_ptr = src + ((size_t)(tile->read_y + r) * padded_stride + tile->read_x) * elem_size;
        void *row_dst_ptr = dst + (size_t)r * (size_t)tile->read_w * elem_size;
        memcpy(row_dst_ptr, row_src_ptr, (size_t)tile->read_w * elem_size);
    }
    return buf;
}

/* Helper: print a neat table of tiles */
static void print_tiles_table(const ProcTile *tiles, int num_procs) {
    printf("Rank | col,row | start_x,start_y | tile_wxh | dW,dH | read_x,y | read_wx h\n");
    printf("-----+---------+-----------------+----------+-------+----------+-----------\n");
    for (int i = 0; i < num_procs; ++i) {
        const ProcTile *t = &tiles[i];
        printf("%4d | %2d,%2d  | %4d,%4d      | %3d x%3d | %2d,%2d | %4d,%4d  | %3d x%3d\n",
            t->rank, t->col, t->row,
            t->start_x, t->start_y,
            t->tile_w, t->tile_h,
            t->dW, t->dH,
            t->read_x, t->read_y,
            t->read_w, t->read_h);
    }
}

/* Helper: ASCII visualization
 * '.' empty, '+' read-only halo, digit (rank%10) compute area (overrides halo)
 */
static void print_ascii_map(const ProcTile *tiles, int num_procs, int padded_w, int padded_h) {
    // allocate map
    char *map = (char*)malloc((size_t)padded_w * padded_h);
    if (!map) return;
    for (int i = 0; i < padded_w * padded_h; ++i) map[i] = '.';

    // mark read areas with '+'
    for (int r = 0; r < num_procs; ++r) {
        const ProcTile *t = &tiles[r];
        if (t->read_w <= 0 || t->read_h <= 0) continue;
        for (int y = t->read_y; y < t->read_y + t->read_h; ++y) {
            for (int x = t->read_x; x < t->read_x + t->read_w; ++x) {
                map[y * padded_w + x] = '+';
            }
        }
    }
    // overwrite compute zones with digit for rank%10
    for (int r = 0; r < num_procs; ++r) {
        const ProcTile *t = &tiles[r];
        if (t->tile_w <= 0 || t->tile_h <= 0) continue;
        char ch = '0' + (char)(t->rank % 10);
        for (int y = t->start_y; y < t->start_y + t->tile_h; ++y) {
            for (int x = t->start_x; x < t->start_x + t->tile_w; ++x) {
                map[y * padded_w + x] = ch;
            }
        }
    }

    // print map with coordinates on left
    printf("\nASCII visualization (rows top->bottom). '.' empty, '+' read/halo, digits compute(rank%%10):\n\n");
    for (int y = 0; y < padded_h; ++y) {
        printf("%3d | ", y);
        for (int x = 0; x < padded_w; ++x) {
            putchar(map[y * padded_w + x]);
        }
        putchar('\n');
    }
    // print x-axis indices (mod 10)
    printf("     ");
    for (int x = 0; x < padded_w; ++x) putchar((char)('0' + (x % 10)));
    putchar('\n');
    free(map);
}

/* NEW: Print the original padded matrix for comparison.
 * Shows each row with index and values (value = y * padded_w + x in this test).
 */
static void print_original_matrix(const int *img, int padded_w, int padded_h) {
    printf("\nOriginal padded matrix (value = y * w + x):\n\n");
    for (int y = 0; y < padded_h; ++y) {
        printf("%3d | ", y);
        for (int x = 0; x < padded_w; ++x) {
            printf("%4d ", img[y * padded_w + x]);
        }
        putchar('\n');
    }
    // x-axis indices
    printf("     ");
    for (int x = 0; x < padded_w; ++x) printf("%4d ", x);
    putchar('\n');
}

/* === main: test driver === */
int main(int argc, char **argv) {
    // Sample parameters (tweak here)
    const int padded_w = 15;   // width (x)
    const int padded_h = 15;   // height (y)
    const int kW = 3;          // kernel width (x)
    const int kH = 3;          // kernel height (y)
    const int num_procs = 5;   // number of ranks

    printf("Image: %dx%d, kernel: %dx%d, procs: %d\n", padded_w, padded_h, kW, kH, num_procs);

    // Create a simple padded image of ints where value = y * padded_w + x
    int *padded_img = (int*)malloc((size_t)padded_w * padded_h * sizeof(int));
    if (!padded_img) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (int y = 0; y < padded_h; ++y) for (int x = 0; x < padded_w; ++x)
        padded_img[y * padded_w + x] = y * padded_w + x;

    // --- NEW: print original matrix before separation ---
    print_original_matrix(padded_img, padded_w, padded_h);

    // Assign tiles
    ProcTile *tiles = assign_tiles_even_spacing(padded_w, padded_h, kW, kH, num_procs);
    if (!tiles) { fprintf(stderr, "assign_tiles failed\n"); free(padded_img); return 1; }

    // Print table
    printf("\nTile assignments:\n");
    print_tiles_table(tiles, num_procs);

    // Print ASCII map
    print_ascii_map(tiles, num_procs, padded_w, padded_h);

    // Demonstrate flattening for each tile: copy and (for small images) print first rows
    printf("\nFlattening demo (each tile's read rectangle copied contiguously):\n");
    for (int r = 0; r < num_procs; ++r) {
        const ProcTile *t = &tiles[r];
        printf("\n-- Rank %d: tile %dx%d at (%d,%d), read %dx%d at (%d,%d)\n",
               t->rank, t->tile_w, t->tile_h, t->start_x, t->start_y,
               t->read_w, t->read_h, t->read_x, t->read_y);

        if (t->read_w <= 0 || t->read_h <= 0) {
            printf("   (no data)\n");
            continue;
        }

        int *buf = (int*)flatten_read_region(padded_img, (size_t)padded_w, t, sizeof(int));
        if (!buf) { printf("   flatten failed\n"); continue; }

        // Show a compact representation of the flattened buffer for smallish regions
        if (t->read_w <= 40 && t->read_h <= 20) {
            for (int y = 0; y < t->read_h; ++y) {
                printf("   ");
                for (int x = 0; x < t->read_w; ++x) {
                    // print value with width 4 for alignment
                    printf("%4d ", buf[y * t->read_w + x]);
                }
                printf("\n");
            }
        } else {
            printf("   (buffer %d x %d copied, too large to print)\n", t->read_w, t->read_h);
        }
        free(buf);
    }

    free(tiles);
    free(padded_img);
    return 0;
}
