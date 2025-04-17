#!/bin/bash

# Script to activate the ml_env Conda environment for WeatherWise

# Define the Conda installation path
CONDA_PATH="/c/Users/kanha/anaconda3"

# Check if Conda path exists
if [ ! -d "$CONDA_PATH" ] || [ ! -f "$CONDA_PATH/Scripts/activate" ]; then
    echo "Error: Conda directory or activation script not found at $CONDA_PATH"
    echo "Please verify your Anaconda installation path (e.g., run 'conda info --base')"
    exit 1
fi

# Initialize Conda for Bash
echo "Initializing Conda from $CONDA_PATH..."
if ! source "$CONDA_PATH/Scripts/activate"; then
    echo "Error: Failed to initialize Conda from $CONDA_PATH/Scripts/activate"
    exit 1
fi

# Check if Conda command is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda command not found after initialization"
    exit 1
fi

# Activate the ml_env environment
echo "Activating ml_env environment..."
conda activate ml_env

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "ml_env" ]; then
    echo "Error: Failed to activate ml_env environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
else
    echo "Successfully activated ml_env environment"
fi