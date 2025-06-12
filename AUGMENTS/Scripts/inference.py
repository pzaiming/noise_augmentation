import os
import argparse
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline
from utils import file_check, path_builder as pb, read_config as rc, write_template as wt, rm_ext
from tqdm import tqdm

class Inference:
    def __init__(self, AUD_PTH, MOD_PTH, CONF_PTH, OUT_PTH, RND):
        self.CWD = os.getcwd() # Instantiate current working directory.
        self.AUD_PTH = pb(self.CWD, file_check(AUD_PTH)) # Perform check, convert relative path to definite path.
        self.MOD_PTH = pb(self.CWD, file_check(MOD_PTH)) # Perform check, convert relative path to definite path.
        self.CONF_PTH = pb(self.CWD, file_check(CONF_PTH)) # Perform check, convert relative path to definite path.
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH)) # Perform check, convert relative path to definite path.
        self.RND = int(RND) if RND is not None else 2
        self.pyannote_conf = None # Instantiate empty pyannote configuration.
        rc(self, self.CONF_PTH) # Read pyannote configuration.
        self.PIPE = self.inst_pyannote_model() # Load PyAnnote pipeline object.
        self.infer() # Perform inference

    def inst_pyannote_model(self):
        model = Model.from_pretrained(self.MOD_PTH) # Accept .ckpt and .bin files.
        pipeline = VoiceActivityDetectionPipeline(segmentation=model)
        pipeline.instantiate(self.pyannote_conf)
        return pipeline

    def infer(self):
        for file in tqdm(os.listdir(self.AUD_PTH)): # For each file in the specified directory...
            for start, end in self.PIPE(pb(self.AUD_PTH, f"{file}")).get_timeline(): # We obtain the timeline using the pipeline...
                wt(pb(self.OUT_PTH, rm_ext(file) + ".txt"), [str(round(float(start), self.RND)), str(round(float(end), self.RND))], "\t") # And write out into a .txt file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", help="Path to a directory containing Audio files.", required=True)
    parser.add_argument("-MOD_PTH", help="Path to a model.", required=True)
    parser.add_argument("-CONF_PTH", help="Path to a model hyperparameter config file.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    parser.add_argument("-RND", help="Decimal place to round off timestamps to.")
    args = parser.parse_args()
    Inference(args.AUD_PTH, args.MOD_PTH, args.CONF_PTH, args.OUT_PTH, args.RND)

if __name__ == "__main__":
    main()  