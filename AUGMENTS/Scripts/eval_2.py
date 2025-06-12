import os
import re
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter 
from utils import path_builder as pb, file_check, rm_ext, write_template as wt, mk_dir
from tqdm import tqdm

class Eval:
    def __init__(self, GND_PTH, HYP_PTH, OUT_PTH):
        self.CWD = os.getcwd() # Instantiate current working directory.
        self.GND_PTH = pb(self.CWD, file_check(GND_PTH)) # Perform check, convert relative path to definite path.
        self.HYP_PTH = pb(self.CWD, file_check(HYP_PTH)) # Perform check, convert relative path to definite path.
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH)) # Perform check, convert relative path to definite path.
        self.eval()

    def eval(self):
        gnd_files = {rm_ext(file) for file in os.listdir(self.GND_PTH)}
        hyp_files = {rm_ext(file) for file in os.listdir(self.HYP_PTH)}
        matching = gnd_files & hyp_files

        # Initialize counters for each state
        duration_counts = {"TP": Counter(), "TN": Counter(), "FP": Counter(), "FN": Counter()}
        true_duration_counts = {"TP": Counter(), "TN": Counter(), "FP": Counter(), "FN": Counter()}

        # Make file folder for individual file results.
        file_dir =  mk_dir(self.OUT_PTH, "FILES")

        for match in tqdm(matching):
            gnd_events = self.extract_events(pb(self.GND_PTH, f"{match}.txt"), "GND")
            hyp_events = self.extract_events(pb(self.HYP_PTH, f"{match}.txt"), "HYP")
            gnd_events_2 = self.extract_events(pb(self.GND_PTH, f"{match}.txt"), "HYP")
            
            # Calculate hyphothesis results and duration.
            merged_events = self.merge_and_sort([gnd_events, hyp_events])
            results, durations = self.compute_states_and_timings(merged_events)
            
            # Calculate ground truth duration.
            true_events = self.merge_and_sort([gnd_events, gnd_events_2])
            _, true_durations = self.compute_states_and_timings(true_events)
            
            # Update counters for each state
            for state, duration_list in durations.items():
                duration_counts[state].update(duration_list)

            for state, duration_list in true_durations.items():
                true_duration_counts[state].update(duration_list)

            # Save results to output files
            [wt(given_pth=pb(file_dir, f"{match}.txt"), template=result, joinby="\t") for result in results]

        self.eval_results(duration_counts, true_duration_counts)

    def extract_events(self, file, event_type):
        events = []
        with open(file, 'r') as f:
            for line in f.readlines():
                match = re.search(r"^(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)$", line)
                if match:
                    start_time, end_time, _ = match.groups()
                    events.append((str(start_time), f"start_{event_type}"))
                    events.append((str(end_time), f"end_{event_type}"))
        return events

    def merge_and_sort(self, event_lists):
        return [event for sublist in event_lists for event in sublist]

    def compute_states_and_timings(self, events):
        results = []
        durations = {"TP": [], "TN": [], "FP": [], "FN": []}

        gnd_speech = False
        hyp_speech = False
        previous_time = None

        for time, label in sorted(events, key=lambda x: float(x[0])):
            if (previous_time is not None) & (previous_time != time):
                duration = round(float(time) - float(previous_time), 2)
                if gnd_speech and hyp_speech:
                    current_state = "TP"
                elif gnd_speech and not hyp_speech:
                    current_state = "FN"
                elif not gnd_speech and hyp_speech:
                    current_state = "FP"
                else:
                    current_state = "TN"
                
                results.append([previous_time, time, current_state])
                durations[current_state].append(duration)

            if "start_GND" in label:
                gnd_speech = True
            elif "end_GND" in label:
                gnd_speech = False
            elif "start_HYP" in label:
                hyp_speech = True
            elif "end_HYP" in label:
                hyp_speech = False

            previous_time = time

        return results, durations

    def eval_results(self, duration_counts, true_duration_counts):
        for state, counter in duration_counts.items():  # Process each state
            hyp_total = round(sum(time * count for time, count in counter.items()), 2) # Calculate hypothesis total
            gnd_total = round(sum(time * count for time, count in true_duration_counts.get(state, {}).items()), 2)  # Ground truth total

            if state == "TP":  # Update totals for True Positive
                hyp_tp = hyp_total
                gnd_tp = gnd_total
            elif state == "TN": # Update totals for True Positive
                hyp_tn = hyp_total
                gnd_tn = gnd_total
            elif state == "FP":  # Update totals for False Positive
                hyp_fp = hyp_total
                gnd_fp = gnd_total
            elif state == "FN":  # Update totals for False Negative
                hyp_fn = hyp_total
                gnd_fn = gnd_total

        summary_template =  [f"Retrieved: {hyp_tp + hyp_fp}", f"Relevant: {gnd_tp}",
        f"Relevant Retrieved: {hyp_tp}", f"Miss: {hyp_fn}",
        f"False Alarm: {hyp_fp}", f"Total: {gnd_tp + gnd_tn + gnd_fp + gnd_fn}",
        f"True Positive: {hyp_tp} / {gnd_tp}",
        f"True Negative: {hyp_tn} / {gnd_tn}",
        f"False Positive: {hyp_fp} / {gnd_fp}",
        f"False Negative: {hyp_fn} / {gnd_fn}"]
        wt(given_pth=pb(self.OUT_PTH, "eval_results.txt"), template=summary_template, joinby="\n")  # Write output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-GND_PTH", help="Path to a directory containing GROUND TRUTH timestamps.", required=True)
    parser.add_argument("-HYP_PTH", help="Path to a directory containing HYPOTHESIS timestamps.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    args = parser.parse_args()
    Eval(args.GND_PTH, args.HYP_PTH, args.OUT_PTH)

if __name__ == "__main__":
    main()  