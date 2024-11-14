
# Parallel Distributed Volume Rendering using MPI with 3D Decomposition

This project implements parallel distributed volume rendering using MPI (Message Passing Interface) for concurrent ray casting along the Z-axis. The dataset is decomposed into 3D subdomains, with each subdomain being processed independently by multiple processes. This approach allows for efficient rendering of large 3D scalar datasets by utilizing the computational power of multiple processors.

## Compilation and Execution Instructions

### Compilation

To compile the code, use the following command:

```bash
mpicxx -g -o try1 main.cpp
```

### Execution

To execute the compiled program, use `mpirun` with the specified number of processes. The syntax for running the program is as follows:

```bash
mpirun -np <num_processes> ./try1 <dataset_file> <procX> <procY> <procZ> <stepSize>
```

For example:

```bash
mpirun -np 4 ./try1 Isabel_1000x1000x200_float32.raw 2 1 2 1
```

### Using a Shell Script

To simplify the execution, create a shell script with the execution command:

1. Create the shell file and add execution permissions:
   ```bash
   chmod +x shell_file.sh
   ```

2. Run the shell file:
   ```bash
   ./shell_file.sh
   ```

## Code Overview

### Key Functions

- **main**: Initializes MPI, parses input arguments, and distributes the dataset to all processes for parallel rendering.

- **readDataset**: Reads and reshapes the 3D scalar dataset file. Uses regex to parse the dataset dimensions from the filename.

- **rayCasting**: Performs ray traversal through each subdomain along the Z-axis. For each (X, Y) coordinate, it accumulates color and opacity values based on transfer functions and interpolates data from the 3D dataset. This function handles communication between adjacent processes along the Z-axis for compositing.

- **volumeRendering**: Manages 3D decomposition of the dataset into subdomains and coordinates ray casting for each subdomain. Composites results across the Z-axis and collects results to form the final output image. Measures computation and communication time for performance analysis.

- **saveImageFromVector**: Saves the composited image in PNG format. This function normalizes pixel data to an 8-bit range for proper image representation.

- **opacityTF** and **colorTF**: Define the transfer functions for opacity and color mapping based on intensity values. These functions handle linear interpolation between defined intensity points.

### Data Flow and Communication

Each process computes its assigned subdomain’s rays. Z-axis processes communicate intermediate opacity values, allowing each to build on the previous section’s compositing result. This ensures that each process can continue compositing where the prior section left off, achieving efficient front-to-back compositing.

## Performance Observations

Performance tests demonstrate that the code efficiently reduces execution time by increasing the number of processes. The optimal configuration depends on both the decomposition dimensions and the step size. Saturation occurs when the core count exceeds 16 due to communication overhead and system constraint.

## Example Test Cases

Test cases are configured as follows:

1. `mpirun -np 8 ./try1 Isabel_1000x1000x200_float32.raw 2 2 2 0.5`
2. `mpirun -np 16 ./try1 Isabel_1000x1000x200_float32.raw 2 2 4 0.5`
3. `mpirun -np 32 ./try1 Isabel_1000x1000x200_float32.raw 2 2 8 0.5`

Execution times and early ray termination ratios are recorded for performance analysis.
