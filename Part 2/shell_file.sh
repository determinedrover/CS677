#!/bin/bash

# Output directory for logs
OUTPUT_DIR="output_logs"
mkdir -p "$OUTPUT_DIR"

# Input file and executable
INPUT_FILE="Isabel_1000x1000x200_float32.raw"
EXECUTABLE="./try1"  # Update if executable name differs

# Ranges for parameters
PROCESS_COUNTS=(1 2 4 8 16)
STEP_SIZES=(0.5 1 2)

# Define multiple (x, y, z) configurations for each process count
declare -A XYZ_CONFIGS
XYZ_CONFIGS[1]="1 1 1"                     # 1 process
XYZ_CONFIGS[2]="2 1 1 1 1 2"               # 2 processes: (2,1,1) and (1,1,2)
XYZ_CONFIGS[4]="2 2 1 4 1 1 1 2 2 1 1 4"         # 4 processes: (2,2,1), (4,1,1), (1,2,2)
XYZ_CONFIGS[8]="2 2 2 4 2 1 2 4 1 2 1 4 8 1 1 1 1 8"         # 8 processes: (2,2,2), (4,2,1), (2,4,1)
XYZ_CONFIGS[16]="4 2 2 2 4 2 2 2 4 8 2 1 2 1 8 16 1 1 1 1 16"        # 16 processes: (4,2,2), (2,4,2), (2,2,4)

# Loop through process counts, step sizes, and predefined x, y, z configurations
for np in "${PROCESS_COUNTS[@]}"; do
    for step_size in "${STEP_SIZES[@]}"; do
        # Get the predefined x, y, z combinations for the current process count
        xyz_combinations="${XYZ_CONFIGS[$np]}"
        
        # Convert xyz_combinations to an array for easy indexing
        xyz_array=($xyz_combinations)
        total_combinations=$((${#xyz_array[@]} / 3))
 
        # Process each (x, y, z) combination for the current process count
        for ((i = 0; i < $total_combinations; i++)); do
            x=${xyz_array[$((i * 3))]}
            y=${xyz_array[$((i * 3 + 1))]}
            z=${xyz_array[$((i * 3 + 2))]}
 
            # Run the mpirun command and store output
            output_file="${OUTPUT_DIR}/output_np${np}_x${x}_y${y}_z${z}_step${step_size}.txt"
            echo "Running: mpirun -np $np $EXECUTABLE $INPUT_FILE $x $y $z $step_size"
            mpirun -np "$np" "$EXECUTABLE" "$INPUT_FILE" "$x" "$y" "$z" "$step_size" > "$output_file"
        done
    done
done
 
echo "All test cases completed. Check the $OUTPUT_DIR directory for output logs."


