CC = mpicc
CFLAGS = -Wall -Wextra -O3 -fopenmp -std=c99
LDFLAGS = -fopenmp -lm

# All source files 
SOURCES = main.c matrix_ops.c file_io.c utils.c convolution.c  conv2d.c
EXECUTABLE = conv2d_app

# Build 
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete! Run with: ./$(EXECUTABLE)"

# Clean
clean:
	rm -f $(EXECUTABLE)
	rm -f *.txt
	@echo "Clean complete"

# Run example
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Help
help:
	@echo "Usage:"
	@echo "  make all    - Build the executable"
	@echo "  make clean  - Remove executable and output files"  
	@echo "  make run    - Build and run with defaults"
	@echo "  make help   - Show this help"

.PHONY: all clean run help