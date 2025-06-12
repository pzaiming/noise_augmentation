import os
import re
import yaml
import argparse
import soundfile as sf
from utils import path_builder as pb, file_check, rm_ext, write_template as wt, check
from tqdm import tqdm

class TimestampRewrite:
    def __init__(self, AUG_PTH, AUD_PTH, GND_PTH, OUT_PTH):
        self.CWD = os.getcwd()
        self.AUG_PTH = pb(self.CWD, file_check(AUG_PTH))
        self.AUD_PTH = pb(self.CWD, file_check(AUD_PTH))
        self.GND_PTH = pb(self.CWD, file_check(GND_PTH))
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH))

        self.compare_lengths()

    def compare_lengths(self):
        for file in tqdm(os.listdir(self.AUD_PTH)):
            aug_file = rm_ext(file) + "_AUG.wav"
            original_file = pb(self.AUD_PTH, file)
            augmented_file = pb(self.AUG_PTH, aug_file)

            if check(augmented_file):
                original_duration = sf.info(original_file).duration
                augmented_duration = sf.info(augmented_file).duration
                self.FACTOR = augmented_duration / original_duration

                gnd_file = pb(self.GND_PTH, rm_ext(file) + ".txt")
                if check(gnd_file):
                    self.write_timestamps(gnd_file, rm_ext(aug_file))

    def read_timestamps(self, file):
        with open(file, 'r') as f:
            start_lst = []
            end_lst = []
            for line in f.readlines():
                match = re.search(r"^(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)$", line)
                if match:
                    start_time, end_time, _ = match.groups()
                    start_lst.append(float(start_time) * self.FACTOR)
                    end_lst.append(float(end_time) * self.FACTOR)
        return start_lst, end_lst

    def write_timestamps(self, gnd_file, base_filename):
        start_lst, end_lst = self.read_timestamps(gnd_file)
        for start, end in zip(start_lst, end_lst):
            wt(pb(self.OUT_PTH, base_filename + ".txt"), [str(start), str(end)], "\t") # And write out into a .txt file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUG_PTH", help="Path to directory containing augmented audio files.", required=True)
    parser.add_argument("-AUD_PTH", help="Path to directory containing original audio files.", required=True)
    parser.add_argument("-GND_PTH", help="Path to directory containing ground truth timestamp files.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to output directory for rewritten timestamp files.", required=True)
    args = parser.parse_args()

    TimestampRewrite(args.AUG_PTH, args.AUD_PTH, args.GND_PTH, args.OUT_PTH)

if __name__ == "__main__":
    main()
