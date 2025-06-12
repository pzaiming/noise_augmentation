import os
import re
import argparse
from collections import defaultdict
from pyannote.core import Annotation, Segment
from utils import path_builder as pb, file_check, write_template as wt, rm_ext
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure, DetectionErrorRate, DetectionAccuracy
from tqdm import tqdm


class Inference:
    def __init__(self, GND_PTH, HYP_PTH, OUT_PTH):
        self.CWD = os.getcwd() # Instantiate current working directory.
        self.GND_PTH = pb(self.CWD, file_check(GND_PTH)) # Perform check, convert relative path to definite path.
        self.HYP_PTH = pb(self.CWD, file_check(HYP_PTH)) # Perform check, convert relative path to definite path.
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH)) # Perform check, convert relative path to definite path.
        self.eval()

    def eval(self):
        gnd_files = [rm_ext(file) for file in os.listdir(self.GND_PTH)]
        hyp_files = [rm_ext(file) for file in os.listdir(self.HYP_PTH)]
        matching = set(gnd_files) & set(hyp_files)
        gnd_lst = self.read_dir(self.GND_PTH, matching)
        hyp_lst = self.read_dir(self.HYP_PTH, matching)
        result = self.compute_eval_metrics(gnd_lst, hyp_lst)
        wt(pb(self.OUT_PTH, "eval_results.txt"), [f"{key}: {str(dic[key])}" for dic in result for key in dic], "\n")    

    def compute_eval_metrics(self, reference_lst, hypothesis_lst):
        dict_lst = [[method.compute_components(reference, hypothesis) for reference, hypothesis in zip(reference_lst, hypothesis_lst)] for method in [DetectionPrecisionRecallFMeasure(), DetectionErrorRate(), DetectionAccuracy()]]
        out_lst = list()
        for metric in dict_lst:
            merged_dict = defaultdict(float)
            for dct in metric:
                for key, value in dct.items():
                    merged_dict[key] += value
            out_lst.append(merged_dict)
        return out_lst
    
    def load_timestamps(self, start_lst, end_lst, speaker):
        inst_obj = Annotation()
        for start, end in zip(start_lst, end_lst):
            inst_obj[Segment(int(start), int(end))] = speaker
        return inst_obj
    
    def return_lst(self, file):
        with open(file, 'r') as f:
            start_lst = list()
            end_lst = list()
            for line in f.readlines():
                match = re.search(r"^(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)$", line)
                if match:
                    start_time, end_time, _ = match.groups()
                    start_lst.append(float(start_time))
                    end_lst.append(float(end_time))
        return start_lst, end_lst

    def read_dir(self, dir_pth, matching_files):
        return_lst = list()
        for file in tqdm(os.listdir(dir_pth)):
            if rm_ext(file) in matching_files:
                start_lst, end_lst = self.return_lst(pb(dir_pth, file))
                obj = self.load_timestamps(start_lst, end_lst, "A")
                return_lst.append(obj)
        return return_lst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-GND_PTH", help="Path to a directory containing GROUND TRUTH timestamps.", required=True)
    parser.add_argument("-HYP_PTH", help="Path to a directory containing HYPOTHESIS timestamps.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    args = parser.parse_args()
    Inference(args.GND_PTH, args.HYP_PTH, args.OUT_PTH)

if __name__ == "__main__":
    main()  