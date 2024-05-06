#!/bin/bash

# Directory containing zip files
SOURCE_DIR="downloads/klines"

# Directory to extract files to
DEST_DIR="data/klines"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all zip files in the source directory
for zip_file in "$SOURCE_DIR"/*.zip; do
    echo "Unzipping $zip_file into $DEST_DIR"
    # Unzip each file to the destination directory
    unzip -o "$zip_file" -d "$DEST_DIR" &
done

echo "All files have been unzipped successfully!"
