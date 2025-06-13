#!/bin/bash
# This script extracts background noise from one audio source and uses it to augment another

# Create directories if they don't exist
mkdir -p ./Noise_Source/
mkdir -p ./Audio/TRAIN/
mkdir -p ./AUG_AUDIO/
mkdir -p ./Extracted_Noise/

# Get the absolute path to the virtual environment's Python
VENV_PYTHON="$(pwd)/.venv/Scripts/python"

# Run the automated noise extraction and augmentation pipeline
echo "Running noise extraction and augmentation..."
if "$VENV_PYTHON" ./Scripts/auto_noise_augment.py -NOISE_SRC ./Noise_Source/ -TARGET ./Audio/TRAIN/ -OUT_PTH ./AUG_AUDIO/ --use-vad; then
    echo "Noise extraction and augmentation complete!"
    echo "Extracted noise files are in ./Extracted_Noise/Noise/"
    echo "Augmented audio files are in ./AUG_AUDIO/"
else
    echo "Error: The Python script failed to execute properly."
    echo "Python used: $VENV_PYTHON"
fi