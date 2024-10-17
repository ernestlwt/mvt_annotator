#!/bin/bash

# Define the folders
folders=(/media/dsta_shared/datasets/fire_academy_1016/data/drone1 /media/dsta_shared/datasets/fire_academy_1016/data/drone2)

# Loop through each folder
for folder in "${folders[@]}"; do
  # Find all .mp4 files in the folder
  find "$folder" -type f -name "*.mp4" | while read -r file; do
    # Extract the base name of the file without the extension
    filename=$(basename "$file" .mp4)
    # Create a directory for the frames if it doesn't exist
    output_dir="${folder}/${filename}_frames"
    mkdir -p "$output_dir"
    # Use ffmpeg to extract all frames
    ffmpeg -i "$file" "$output_dir/frame_%06d.png"
  done
done
