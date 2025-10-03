#!/bin/bash

# Batch processing script for HDF5 demonstration files
# This script finds all HDF5 files in the data/raw_hdf5 directory
# and processes them one by one using the replay_hdf5.py script

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Directory containing HDF5 files
HDF5_DIR="$PROJECT_ROOT/data/raw_hdf5"

# Path to the replay script
REPLAY_SCRIPT="$PROJECT_ROOT/scripts/helpers/replay_hdf5.py"

# Output directory (default location)
OUTPUT_DIR="$PROJECT_ROOT/data/replayed_activities"

# Check if the HDF5 directory exists
if [ ! -d "$HDF5_DIR" ]; then
    echo "Error: Directory $HDF5_DIR does not exist"
    exit 1
fi

# Check if the replay script exists
if [ ! -f "$REPLAY_SCRIPT" ]; then
    echo "Error: Replay script $REPLAY_SCRIPT does not exist"
    exit 1
fi

# Find all HDF5 files in the directory
echo "Searching for HDF5 files in: $HDF5_DIR"
HDF5_FILES=($(find "$HDF5_DIR" -maxdepth 1 -name "*.hdf5" -type f))

if [ ${#HDF5_FILES[@]} -eq 0 ]; then
    echo "No HDF5 files found in $HDF5_DIR"
    exit 1
fi

echo "Found ${#HDF5_FILES[@]} HDF5 files to process"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Process each file individually
for i in "${!HDF5_FILES[@]}"; do
    FILE="${HDF5_FILES[$i]}"
    echo ""
    echo "========================================="
    echo "Processing file $((i+1))/${#HDF5_FILES[@]}: $(basename "$FILE")"
    echo "========================================="
    
    # Run the replay script with the current file
    python "$REPLAY_SCRIPT" --file "$FILE" --output_dir "$OUTPUT_DIR"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $(basename "$FILE")"
    else
        echo "Error processing: $(basename "$FILE")"
        echo "Continuing with next file..."
    fi
    
    echo "Completed file $((i+1))/${#HDF5_FILES[@]}"
    # Sleep for 10 seconds before processing next file
    echo "Waiting 10 seconds before next file..."
    sleep 10
done

echo ""
echo "========================================="
echo "All files processed!"
echo "========================================="

