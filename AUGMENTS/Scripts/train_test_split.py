import os
import re
import shutil
import argparse
from sklearn.model_selection import train_test_split
from utils import path_builder as pb, file_check, write_template as wt, rm_ext


class TTS:
    def __init__(self, AUD_PTH, OUT_PTH, train_ratio=0.7, dev_ratio=0.1, test_ratio=0.2):
        self.CWD = os.getcwd() # Instantiate current working directory.
        self.AUD_PTH = pb(self.CWD, file_check(AUD_PTH)) # Perform check, convert relative path to definite path.
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH)) # Perform check, convert relative path to definite path.
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.split()

    def split(self):
        """Split audio files into Train, Dev, and Test directories."""
        # Get all audio files
        audio_files = [f for f in os.listdir(self.AUD_PTH) if os.path.isfile(os.path.join(self.AUD_PTH, f))]
        if not audio_files:
            raise ValueError("No audio files found in the specified directory.")

        # Split Train + Remaining first
        train_files, remaining_files = train_test_split(audio_files, test_size=1 - self.train_ratio, random_state=42)

        # Split Remaining into Dev and Test
        test_size = self.test_ratio / (self.test_ratio + self.dev_ratio)  # Proportion of remaining files for Test
        dev_files, test_files = train_test_split(remaining_files, test_size=test_size, random_state=42)

        # Create output directories and copy files
        self.copy_files(train_files, os.path.join(self.OUT_PTH, "TRAIN"))
        self.copy_files(dev_files, os.path.join(self.OUT_PTH, "DEV"))
        self.copy_files(test_files, os.path.join(self.OUT_PTH, "TEST"))

        print(f"Files successfully split:")
        print(f"TRAIN: {len(train_files)} files -> {os.path.join(self.OUT_PTH, 'TRAIN')}")
        print(f"DEV: {len(dev_files)} files -> {os.path.join(self.OUT_PTH, 'DEV')}")
        print(f"TEST: {len(test_files)} files -> {os.path.join(self.OUT_PTH, 'TEST')}")

    def copy_files(self, file_list, target_dir):
        """Copy files to the target directory."""
        os.makedirs(target_dir, exist_ok=True)
        for file in file_list:
            shutil.copy(os.path.join(self.AUD_PTH, file), target_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", help="Path to a directory containing GROUND TRUTH timestamps.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    args = parser.parse_args()
    TTS(args.AUD_PTH, args.OUT_PTH)

if __name__ == "__main__":
    main()  