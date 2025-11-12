# 2D Convolution with MPI + OpenMP
## High-Performance Computing Implementation

**Author:** Reuben Stanley  
**Student ID:** 23775365  
**Unit:** CITS5507 - High Performance Computing  
**Institution:** University of Western Australia  
**Semester:** 2024

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Memory Layout Architecture](#memory-layout-architecture)
3. [2D Domain Decomposition Strategy](#2d-domain-decomposition-strategy)
4. [Design Justifications](#design-justifications)
5. [Quick Start Guide](#quick-start-guide)
6. [Running Experiments](#running-experiments)
7. [Performance Characteristics](#performance-characteristics)
8. [Implementation Highlights](#implementation-highlights)

---

## Overview

This project implements high-performance 2D convolution using a hybrid MPI+OpenMP approach with:
- **1D contiguous memory layout** for cache efficiency
- **2D Cartesian domain decomposition** for load balancing
- **Four-directional halo exchange** for minimal communication overhead
- **Non-blocking MPI** for communication/computation overlap

**Key Performance Metrics:**
- Cache miss rate: **1.48%** (exceptional)
- Sustained throughput: **3.7×10¹⁰ FLOP/s**
- Strong scaling: **25.8× speedup** on 32 workers (81% efficiency)
- Memory efficiency: **94% savings** vs broadcast approach

---

## Memory Layout Architecture

### 1D Contiguous Array Design

Each MPI process maintains a **single contiguous 1D array** storing its local tile with surrounding halo regions:

```c
// Memory allocation per process
int padded_width = tile_width + 2 × halo_width;
int padded_height = tile_height + 2 × halo_height;
float *local_data = malloc(padded_width × padded_height × sizeof(float));

// Element access with row-major indexing
element[row][col] = local_data[(row + halo_height) × padded_width + (col + halo_width)];
```

### Memory Structure

```
┌─────────────────────────────────┐
│     TOP HALO (from above)       │  ← halo_height rows
├──┬──────────────────────────┬───┤
│L │                          │ R │
│E │   LOCAL COMPUTE REGION   │ I │  ← tile_height rows
│F │   (assigned tile)        │ G │
│T │                          │ H │
│  │                          │ T │
├──┴──────────────────────────┴───┤
│   BOTTOM HALO (from below)      │  ← halo_height rows
└─────────────────────────────────┘
   ↑                             ↑
   halo_width              halo_width
   columns                 columns
```

### Why 1D Contiguous Layout?

**Advantages:**

1. ✅ **No pointer indirection** - Single level of addressing
2. ✅ **Cache-friendly** - Perfect spatial locality for row-major traversal
3. ✅ **SIMD vectorization** - Compiler can auto-vectorize inner loops
4. ✅ **Predictable access** - Hardware prefetcher works optimally
5. ✅ **Reduced overhead** - Eliminates pointer array (saves `height × 8` bytes)

**Result:** Achieved **1.48% cache miss rate** compared to typical 3-5% for 2D pointer arrays.

---

## 2D Domain Decomposition Strategy

### Cartesian Process Grid

Processes are arranged in a 2D grid using `MPI_Cart_create`:

```
Example: 4 processes in 2×2 grid

┌─────────┬─────────┐
│ Proc 0  │ Proc 1  │  prow=0
│ (0,0)   │ (0,1)   │
├─────────┼─────────┤
│ Proc 2  │ Proc 3  │  prow=1
│ (1,0)   │ (1,1)   │
└─────────┴─────────┘
   pcol=0    pcol=1
```

### Work Distribution

Each process receives a rectangular tile:

```c
// Compute output tile dimensions per process
base_rows = output_height / num_row_procs;
base_cols = output_width / num_col_procs;

// Distribute remainder evenly
my_rows = base_rows + (my_prow < rem_rows ? 1 : 0);
my_cols = base_cols + (my_pcol < rem_cols ? 1 : 0);

// Compute required input dimensions (accounting for stride and kernel)
input_rows_needed = (my_rows - 1) × stride_h + kernel_h;
input_cols_needed = (my_cols - 1) × stride_w + kernel_w;
```

### Four-Directional Halo Exchange

Each process communicates with up to **4 neighbors**:

```c
MPI_Request requests[8];
int req_count = 0;

// Post receives first (avoid deadlock)
if (has_top_neighbor)
    MPI_Irecv(top_halo, size, MPI_FLOAT, top_nbr, TAG, comm, &requests[req_count++]);
if (has_bottom_neighbor)
    MPI_Irecv(bottom_halo, size, MPI_FLOAT, bot_nbr, TAG, comm, &requests[req_count++]);
if (has_left_neighbor)
    MPI_Irecv(left_halo, size, MPI_FLOAT, left_nbr, TAG, comm, &requests[req_count++]);
if (has_right_neighbor)
    MPI_Irecv(right_halo, size, MPI_FLOAT, right_nbr, TAG, comm, &requests[req_count++]);

// Post sends (can proceed immediately)
if (has_top_neighbor)
    MPI_Isend(top_boundary, size, MPI_FLOAT, top_nbr, TAG, comm, &requests[req_count++]);
// ... (similarly for bottom, left, right)

// Wait for all communications
MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
```

**Key features:**
- **Non-blocking** - Enables communication/computation overlap
- **Simultaneous** - All 4 directions exchange at once
- **Bidirectional** - Full-duplex communication on all links

---

## Design Justifications

### Question 1: Why Not Broadcast the Full Matrix?

**Answer: Memory and Bandwidth Efficiency**

Broadcasting would:
- ❌ Use `num_procs × matrix_size` memory (massive waste)
- ❌ Create bandwidth bottleneck at root
- ❌ Not scale beyond single-node shared memory

**Example (20000×20000 matrix, 16 processes):**
```
Broadcast approach:  16 × 1.6 GB = 25.6 GB total
Halo exchange:       16 × 100 MB = 1.6 GB total
Savings:             24 GB (94% reduction!)
```

### Question 2: Why 2D Decomposition Instead of 1D?

**Answer: Superior Cache Locality and Communication Balance**

**Comparison:**

| Metric | 1D (Row-Only) | 2D (Row & Column) | Advantage |
|--------|---------------|-------------------|-----------|
| Tile size | 5000×20000 = 400 MB | 10000×10000 = 100 MB | **4× smaller** |
| L3 cache fit | ❌ No | ⚠️ Partial | **Better** |
| Perimeter | 50,000 cells | 40,000 cells | **20% less** |
| Neighbors | 2 (top/bottom) | 4 (all sides) | **2× links** |
| Comm per direction | 240 KB | 120 KB | **Lower latency** |
| Load balance | Limited | Flexible | **√P × √P grids** |

**Result:** 2D achieves **1.48% cache miss rate** vs typical 3-5% for 1D.

### Question 3: Why Halo Exchange Instead of Ghost Cell Replication?

**Answer: Zero Redundant Computation**

**Halo exchange (our choice):**
```
Process 0 computes: rows 0-4999 (no overlap)
Process 1 computes: rows 5000-9999 (no overlap)
Boundary data shared via MPI communication
Result: Each output cell computed EXACTLY ONCE
```

**Ghost cell replication (alternative):**
```
Process 0 computes: rows 0-5002 (overlap at boundary)
Process 1 computes: rows 4998-9999 (overlap at boundary)
                      ↑↑↑↑↑
                   Rows 4998-5002: Computed TWICE (redundant!)
```

**Benefits:**
- ✅ Zero redundant FLOPs
- ✅ Cleaner semantics (each process owns its tile)
- ✅ No deduplication logic needed

---

## Quick Start Guide

### Prerequisites

```bash
# Ensure MPI and OpenMP are available
module load gcc/12.2.0        # Or your system's compiler
module load openmpi/4.1.5     # Or your MPI implementation

# Python for analysis (optional)
pip install pandas matplotlib numpy
```

### Compilation

```bash
# Build the executable
make clean && make

# Verify compilation
ls -lh conv2d_app
```

**Compilation flags used:**
```makefile
CC = mpicc
CFLAGS = -O3 -fopenmp -march=native -mtune=native -ftree-vectorize
LDFLAGS = -fopenmp -lm
```

### Basic Usage

```bash
# Serial execution (baseline)
./conv2d_app -W 1000 -H 1000 -kW 3 -kH 3 -sW 1 -sH 1 -algo naive

# OpenMP parallel (4 threads)
./conv2d_app -W 1000 -H 1000 -kW 3 -kH 3 -sW 1 -sH 1 -algo parallel

# Hybrid MPI+OpenMP (4 processes × 4 threads)
mpirun -np 4 ./conv2d_app -W 1000 -H 1000 -kW 3 -kH 3 -sW 1 -sH 1 -algo hybrid

# With stride (downsampling)
mpirun -np 4 ./conv2d_app -W 1000 -H 1000 -kW 3 -kH 3 -sW 2 -sH 2 -algo hybrid

# Save output to file
mpirun -np 4 ./conv2d_app -W 1000 -H 1000 -kW 3 -kH 3 -o output.txt
```

### Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `-W` | Input width | `-W 10000` |
| `-H` | Input height | `-H 10000` |
| `-kW` | Kernel width | `-kW 7` |
| `-kH` | Kernel height | `-kH 7` |
| `-sW` | Stride width | `-sW 2` |
| `-sH` | Stride height | `-sH 2` |
| `-algo` | Algorithm: `naive`, `parallel`, `hybrid` | `-algo hybrid` |
| `-r` | Random initialization | `-r` |
| `-i` | Input file | `-i input.txt` |
| `-ki` | Kernel file | `-ki kernel.txt` |
| `-o` | Output file | `-o result.txt` |
| `-t` | Time measurement | `-t` |

---

## Running Experiments

### Local Execution

```bash
# Make scripts executable
chmod +x run_experiments.sh

# Run all experiments (~30-60 minutes)
./run_experiments.sh

# Results saved in: experiment_results/
```

**Experiments included:**
1. **Algorithm comparison** - Serial, OpenMP, various hybrid configs
2. **Stride effect** - How stride affects performance
3. **Thread scaling** - Optimal number of threads
4. **Stress test** - Maximum problem size (optional, ~60 min)
5. **Cache analysis** - Requires `perf` tool (Linux only)

### HPC Execution (Setonix/Kaya)

```bash
# Edit Slurm script with your account
nano run_experiments_slurm.sh
# Change: #SBATCH --account=<YOUR_ACCOUNT>

# Submit batch job
sbatch run_experiments_slurm.sh

# Monitor job
squeue -u $USER
watch -n 5 'squeue -u $USER'

# View results when complete
cat experiments_*.out
```

**Slurm configuration:**
```bash
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
```

### Analyzing Results

```bash
# Generate plots and statistics
python3 analyze_results.py experiment_results/

# View generated figures
ls experiment_results/*.png

# Read summary report
cat experiment_results/summary_report.txt
```

**Generated outputs:**
- `algorithm_comparison.png` - FLOP/s and Speedup plots
- `stride_effect.png` - Stride analysis
- `thread_scaling.png` - Thread scaling curve
- `summary_report.txt` - Statistical summary

---

## Performance Characteristics

### Expected Results

Based on 20000×20000 matrix with 7×7 kernel:

**Algorithm Comparison (stride 2×2):**

| Configuration | Time (s) | Speedup | Efficiency |
|--------------|----------|---------|------------|
| Serial | 3.709 | 1.0× | 100% |
| OpenMP (4T) | 0.822 | 4.5× | 113% |
| Hybrid (4×4, 2N) | 0.318 | 11.7× | 73% |
| Hybrid (8×4, 2N) | 0.144 | 25.8× | 81% |

**Stride Effect (4×4 OpenMP, 2 nodes):**

| Stride | Output Size | Time (s) | FLOP/s |
|--------|-------------|----------|---------|
| 1×1 | 19994×19994 | 1.058 | 3.70×10¹⁰ |
| 2×2 | 9998×9998 | 0.259 | 3.78×10¹⁰ |
| 4×4 | 5000×5000 | 0.064 | 3.83×10¹⁰ |
| 8×8 | 2501×2501 | 0.017 | 3.61×10¹⁰ |

**Key observation:** FLOP/s remains constant (~3.7×10¹⁰) across all strides, indicating **compute-bound performance** with negligible communication overhead.

**Thread Scaling (20000×20000, stride 1×1):**

| Threads | Time (s) | Speedup | Efficiency |
|---------|----------|---------|------------|
| 1 | 3.839 | 1.0× | 100% |
| 4 | 1.211 | 3.2× | 79% |
| 8 | 1.047 | 3.7× | 46% |
| 12 | 1.042 | 3.7× | 31% |
| 16+ | ~1.04 | 3.7× | <25% |

**Optimal configuration:** 8-12 threads per node

### Cache Performance

Measured using `perf stat` on 10000×10000 matrix:

```
Cache references:  65,895,590
Cache misses:         976,625
Miss rate:              1.48%
```

**Why 1.48% is exceptional:**
- Typical 2D pointer arrays: 3-5% miss rate
- **3.4× better cache efficiency**
- Achieved through:
  1. Contiguous 1D memory layout
  2. Small 2D tiles (better cache fit)
  3. Sequential kernel scanning
  4. Minimal halo overhead

---

## Implementation Highlights

### Code Structure

```
src/
├── main.c              # Entry point, CLI parsing
├── conv2d.c            # Main convolution with MPI+OpenMP
├── convolution.c       # Convolution kernel computation
├── matrix_ops.c        # Matrix utilities (padding, generation)
├── file_io.c           # I/O operations (MPI collective writes)
└── utils.c             # Helper functions

include/
├── conv2d.h
├── convolution.h
├── matrix_ops.h
├── file_io.h
└── utils.h
```

### Key Algorithms

**Convolution kernel (OpenMP parallelized):**
```c
#pragma omp parallel for collapse(2) schedule(static)
for (int r = 0; r < my_out_rows; ++r) {
    for (int c = 0; c < my_out_cols; ++c) {
        float acc = 0.0f;
        int in_r = r * stride_h;
        int in_c = c * stride_w;
        
        // Kernel convolution
        for (int kr = 0; kr < kernel_h; ++kr) {
            float *in_row_ptr = local_data + (in_r + kr) * padded_width + in_c;
            float *k_row_ptr = kernel + kr * kernel_w;
            
            #pragma omp simd reduction(+:acc)
            for (int kc = 0; kc < kernel_w; ++kc) {
                acc += in_row_ptr[kc] * k_row_ptr[kc];
            }
        }
        
        local_output[r * my_out_cols + c] = acc;
    }
}
```

**Optimizations:**
- ✅ Loop collapse for better parallelization
- ✅ SIMD reduction for inner loop
- ✅ Pointer arithmetic to reduce indexing overhead
- ✅ Static scheduling for load balance

### Stride Handling

**Key insight:** With stride, output dimensions differ from input:

```c
// Calculate output dimensions
int out_H = (H + pad_H - kernel_H) / stride_H + 1;
int out_W = (W + pad_W - kernel_W) / stride_W + 1;

// Partition OUTPUT space across processes (not input!)
int base_rows = out_H / num_row_procs;
int base_cols = out_W / num_col_procs;

// Compute required INPUT dimensions per process
int input_rows_needed = (my_out_rows - 1) * stride_H + kernel_H;
int input_cols_needed = (my_out_cols - 1) * stride_W + kernel_W;
```

**Critical:** Partitioning must be based on **output dimensions** to ensure correct work distribution.

### Verification

Test correct stride implementation:

```bash
# Stride 1×1 (baseline)
./conv2d_app -W 100 -H 100 -kW 3 -kH 3 -sW 1 -sH 1 -o test1.txt
head -1 test1.txt  # Should show: 100 100

# Stride 2×2 (downsampling by 2)
./conv2d_app -W 100 -H 100 -kW 3 -kH 3 -sW 2 -sH 2 -o test2.txt
head -1 test2.txt  # Should show: 50 50

# Stride 3×3 (downsampling by 3)
./conv2d_app -W 99 -H 99 -kW 5 -kH 5 -sW 3 -sH 3 -o test3.txt
head -1 test3.txt  # Should show: 33 33
```

---

## Troubleshooting

### Common Issues

**Problem:** `conv2d_app: command not found`
```bash
# Solution: Ensure it's compiled and in current directory
make clean && make
ls -l conv2d_app
./conv2d_app --help
```

**Problem:** `mpirun: command not found`
```bash
# Solution: Load MPI module (HPC) or install locally
module load openmpi         # On HPC
brew install open-mpi       # On macOS
sudo apt install openmpi-bin # On Ubuntu/Debian
```

**Problem:** Wrong output dimensions with stride
```bash
# Solution: Verify stride bug is fixed
./conv2d_app -W 100 -H 100 -kW 3 -kH 3 -sW 2 -sH 2 -o test.txt
head -1 test.txt  # Must show: 50 50 (not 100 100)
```

**Problem:** Low performance / high cache misses
```bash
# Solution: Check compiler optimizations
make clean
make CFLAGS="-O3 -march=native -ftree-vectorize -fopt-info-vec"

# Verify vectorization
# Should see messages like: "loop vectorized"
```

**Problem:** Out of memory on large problems
```bash
# Solution: Increase Slurm memory allocation
#SBATCH --mem=128G  # Instead of 64G

# Or reduce problem size
./conv2d_app -W 10000 -H 10000  # Instead of 20000×20000
```

---

## Performance Tuning Tips

### Optimal Configuration

For best performance on most HPC systems:

```bash
# Use power-of-2 process counts for 2D grids
# Good: 4 (2×2), 16 (4×4), 64 (8×8)
# Avoid: 6, 10, 12 (awkward 2D factorization)

# Set OpenMP threads to physical cores per node
export OMP_NUM_THREADS=4  # Adjust to your system

# Enable thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Run with optimal config
mpirun -np 16 ./conv2d_app -W 20000 -H 20000 -kW 7 -kH 7 -algo hybrid
```

### Compiler Optimization

```bash
# Maximum optimization
CFLAGS="-O3 -march=native -mtune=native -ftree-vectorize -funroll-loops"

# With vectorization report (to verify SIMD usage)
CFLAGS="-O3 -march=native -fopt-info-vec"

# Debug version (for development)
CFLAGS="-O0 -g -Wall -Wextra"
```

---

## References

- Course materials: CITS5507 High Performance Computing
- MPI documentation: https://www.mpich.org/documentation/
- OpenMP specification: https://www.openmp.org/specifications/
- Setonix HPC user guide: https://pawsey.org.au/systems/setonix/

---

## Acknowledgments

This implementation was developed as part of CITS5507 High Performance Computing coursework at the University of Western Australia. The design combines theoretical concepts from the course with practical optimization techniques for modern HPC systems.

**Special thanks to:**
- Course instructors and tutors
- Pawsey Supercomputing Centre for access to Setonix
- Project partner Erica Kong for collaboration on report analysis

---

## License

This project is submitted as academic coursework for CITS5507. Usage should comply with UWA academic integrity policies.

---

**Last Updated:** November 2024  
**Contact:** Reuben Stanley (Student ID: 23775365)  
**Repository:** [Private - Academic Submission]

---

*This README provides comprehensive documentation for the 2D convolution implementation. For additional details on experimental methodology and performance analysis, refer to the full assignment report.*
