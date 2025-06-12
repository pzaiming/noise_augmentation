import os
import yaml
import torch
import argparse
import speechbrain as sb
import speechbrain.augment.time_domain as time_augmentations
from speechbrain.augment.augmenter import Augmenter
from speechbrain.dataio.dataio import read_audio, write_audio
from utils import path_builder as pb, file_check, mk_dir, rm_ext
from tqdm import tqdm

class AugmentPipeline:
    def __init__(self, AUD_PTH, CONF_PTH, OUT_PTH):
        self.CWD = os.getcwd() # Instantiate current working directory.
        self.AUD_PTH = pb(self.CWD, file_check(AUD_PTH))
        self.CONF_PTH = pb(self.CWD, file_check(CONF_PTH))
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH))
        
        augments, arg_list = self.instantiate_augments()
        self.load_augmenter_class(augments)
        self.augment_data(arg_list)

    def instantiate_augments(self) -> list:
        with open(self.CONF_PTH, "r") as config_file:
            config_data = yaml.safe_load(config_file)
        augments = []  # Store initialized augmentations here
        self.AUG_CONF = None
        arg_list = []
        for key, value in config_data.items():
            if key == "Augmenter":
                self.AUG_CONF = value
                continue
            elif key == "TimeDomain":
                for augment_name, args in value.items():
                    arg_list.append(augment_name)
                    try: # Dynamically get each augment and their arguments from the module
                        augment_class = getattr(time_augmentations, augment_name)
                    except AttributeError:
                        raise ValueError(f"Augmentation {augment_name} not found in the module.")
                    if isinstance(args, dict): # Initialize the augmentation with the arguments
                        augments.append(augment_class(**args))
                    else:
                        raise ValueError(f"Invalid arguments for {augment_name}: {args}")
        expected_config = {
        "parallel_augment": True,
        "concat_original": False,
        "shuffle_augmentations": False
        }
        if "Augmenter" in config_data:
            augmenter_config = config_data["Augmenter"]
            is_match = all(
                augmenter_config.get(key) == value for key, value in expected_config.items()
            )
            if is_match:
                self.INDIVCHECK = True
            else:
                self.INDIVCHECK = False
        else:
            raise ValueError("'Augmenter' section not found in the configuration file.")
        return augments, arg_list

    def load_augmenter_class(self, augment_lst):
        if isinstance(self.AUG_CONF, dict):
            self.AUG = Augmenter(**self.AUG_CONF, augmentations=augment_lst)
        else:
            raise ValueError(f"Invalid arguments for Augmenter: {self.AUG_CONF}")

    def augment_data(self, arg_list):
        for file in tqdm(os.listdir(self.AUD_PTH)):
            signal = read_audio(f'{pb(self.AUD_PTH, file)}')
            clean = signal.unsqueeze(0)
            augmented_signal, lenghts = self.AUG(clean, lengths=torch.ones(1))

            num_signals = augmented_signal.shape[0] # WARNING!!! Dependng on arguments passed, augmented_signal may be of a larger shape than 1.
            if num_signals > 1:
                idx = 0
                for signal in augmented_signal:
                    if self.INDIVCHECK:
                        signal_dir = mk_dir(self.OUT_PTH, arg_list[idx])
                    else:
                        signal_dir = mk_dir(self.OUT_PTH, str(idx))
                    write_audio(pb(signal_dir, f"{rm_ext(file)}_AUG.wav"), signal, samplerate=16000)
                    idx += 1
            else:
                write_audio(pb(self.OUT_PTH, f"{rm_ext(file)}_AUG.wav"), augmented_signal[0], samplerate=16000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", help="Path to a directory containing audio files to be augmented.", required=True)
    parser.add_argument("-CONF_PTH", help="Path to a config file that contains all the arguments required for each augmentation.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    args = parser.parse_args()
    AugmentPipeline(args.AUD_PTH, args.CONF_PTH, args.OUT_PTH)

if __name__ == "__main__":
    main()  