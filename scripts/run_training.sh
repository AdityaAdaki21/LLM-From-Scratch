#!/bin/bash

echo "Starting LLM training script..."

# Activate your virtual environment if you have one
# source venv/bin/activate

# Set environment variables if needed (e.g., CUDA devices)
# export CUDA_VISIBLE_DEVICES=0

# Run the main training script
# Make sure the paths in src/config.py are correct relative to the project root
python src\\train.py

# Check the exit code
if [ $? -eq 0 ]; then
  echo "Training script finished successfully."
else
  echo "Training script failed with error code $?."
fi

# Deactivate virtual environment (optional)
# deactivate

echo "Script finished."