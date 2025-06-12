import os
import re
import json
import yaml
import shutil
import argparse
import pandas as pd
import torchaudio
from pathlib import Path
import pytorch_lightning as pl
from collections import defaultdict
from pyannote.audio import Model
from pyannote.pipeline import Optimizer
from pyannote.database import registry, FileFinder
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.tasks import VoiceActivityDetection
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline
from utils import path_builder as pb, file_check, write_template as wt, mk_file, rm_ext, mk_dir, read_config as rc, check

class Trainer:
    def __init__(self, AUD_PTH, GND_PTH, BASE_MOD_PTH, TRAIN_CONF_PTH, OUT_PTH):
        self.CWD = os.getcwd() # Instantiate current working directory.
        self.GND_PTH = pb(self.CWD, file_check(GND_PTH)) # Perform check, convert relative path to definite path.
        self.AUD_PTH = pb(self.CWD, file_check(AUD_PTH)) # Perform check, convert relative path to definite path.
        self.BASE_MOD_PTH = pb(self.CWD, file_check(BASE_MOD_PTH)) # Perform check, convert relative path to definite path.
        self.TRAIN_CONF_PTH = pb(self.CWD, file_check(TRAIN_CONF_PTH)) # Perform check, convert relative path to definite path.
        self.OUT_PTH = pb(self.CWD, file_check(OUT_PTH)) # Perform check, convert relative path to definite path.
        self.WORK = pb(self.CWD, "work", type="DIR")  # Temporary directory to store files required for training.
        
        # Call functions
        self.write()
        self.train()

    def write(self):
        """
        Write necessary files for training, including URI, RTTM, UEM files, and the PyAnnote protocol YAML file.
        """
        # Instantiate paths required for training.
        uri_pth = pb(self.WORK, "URI", type="DIR")  # PyAnnote URI directory, stores files listing the train and dev files.
        uem_pth = pb(self.WORK, "UEM", type="DIR")  # PyAnnote UEM directory, contains information for file and scoring zone.
        rttm_pth = pb(self.WORK, "RTTM", type="DIR")  # PyAnnote RTTM directory, contains information for file and segment timings.
        self.TRAIN = pb(self.AUD_PTH, "TRAIN")  # Path to /Audio/Train with training files.
        self.DEV = pb(self.AUD_PTH, "DEV")  # Path to /Audio/Dev with validation files.

        # Build URI
        print("Building URI")
        train_lst = [rm_ext(file) for file in os.listdir(self.TRAIN)]  # Files that are in the train set.
        wt(mk_file(uri_pth, "train"), train_lst, f"\n")  # Creating train URI file.
        dev_lst = [rm_ext(file) for file in os.listdir(self.DEV)]  # Files that are in the dev set.
        wt(mk_file(uri_pth, "development"), dev_lst, f"\n")  # Creating dev URI file.

        # Build RTTM dataframes containing RTTM data such as timestamps, etc
        print("Building RTTM")
        train_df = pd.concat([self.read_segments(file, "train") for file in train_lst])  # Building train dataframe for writing into RTTM files.
        dev_df = pd.concat([self.read_segments(file, "development") for file in dev_lst])  # Building dev dataframe for writing into RTTM files.
        
        # Build RTTM
        self.write_rttm(train_df, "train", rttm_pth)  # Writing RTTM files for train files.
        self.write_rttm(dev_df, "development", rttm_pth)  # Writing RTTM files for dev files.

        # Build UEM
        print("Building UEM")
        self.write_uem(train_df, "train", uem_pth)  # Writing UEM files for train files.
        self.write_uem(dev_df, "development", uem_pth)  # Writing UEM files for dev files.

        # Build yaml file
        print("Building yml")
        self.write_yml(["train", "development"], [uri_pth, rttm_pth, uem_pth])  # Writing protocol for PyAnnote database.

    def write_rttm(self, dct: pd.DataFrame, folder: str, pth: str):
        """
        Writes a series of start times, durations for a series of files into .rttm files loosely following the .rttm template:
            Type: Segment type. Should always be <SPEAKER>.
            File ID: File name. Basename of the recording minus extension.
            Channel ID: Channel (1-indexed) that turn is on. Should always be <1>.
            Turn Onset: Onset of turn in seconds from beginning of recording.
            Turn Duration: Duration of turn in seconds.
            Orthography Field: Should always be <NA>.
            Speaker Type: Should always be <NA>.
            Speaker Name: Name or ID of speaker of turn. Should be unique within scope of each file <ID>.
            Confidence Score: System confidence (probability) that information is correct. Should always be <NA>.
            Signal Lookahead Time: Should always be <NA>.

        Args:
            dct (pd.DataFrame): Full dictionary with the data.
            folder (str): Train, Dev or Test.
            pth (str): Path to RTTM folder.
        """
        for name, onset_lst, duration_lsts in zip(dct["Name"], dct["Start"], dct["Duration"]):
            for onset, duration in zip(onset_lst, duration_lsts):
                template = ['SPEAKER', name, '1', str(onset), str(duration), '<NA>', '<NA>', 'DUMSPKR', '<NA>', '<NA>']
                wt(mk_file(mk_dir(pth, folder), name, ".rttm"), template, ' ')

    def read_segments(self, file: str, typ: str) -> pd.Series:
        """
        Dataframe series builder function to read and organize data into series for concatenation into a dataframe.

        Args:
            file (str): Name of the file to be read.
            typ (str): 'train' or 'development' to locate the directory the file is stored in.

        Returns:
            pd.Series: Series containing Name, Start, Duration, and Length information.
        """
        if typ == "train":
            base_pth = self.TRAIN
        else:
            base_pth = self.DEV
        waveform, sample_rate = torchaudio.load(pb(base_pth, file + ".wav"))
        length = len(waveform[0]) / sample_rate
        with open(pb(self.GND_PTH, file + ".txt"), "r") as f:
            start_lst = []
            duration_lst = []
            for line in f.readlines():
                match = re.search(r"^(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)$", line)
                if match:
                    start_time, end_time, _ = match.groups()
                    start_lst.append(float(start_time))
                    duration_lst.append(float(end_time) - float(start_time))
            return pd.Series({"Name": file, "Start": start_lst, "Duration": duration_lst, "Length": length}).to_frame().T

    def write_uem(self, dct: pd.DataFrame, folder: str, pth: str):
        """
        Writes the length and name of each file into .uem files following the .uem template:
            File ID: File name. Basename of the recording minus extension.
            File Onset: Timestamp of the beginning of the file. Should always be <0.00>
            File Length: Duration of the file in seconds.
        
        Args:
            dct (pd.DataFrame): Full dictionary with the data.
            folder (str): Train, Dev or Test.
            pth (str): Path to UEM folder.
        """
        for name, length in zip(dct["Name"], dct["Length"]):
            template = [name, "1", "0.00", str(length)]
            wt(mk_file(mk_dir(pth, folder), name, ".uem"), template, ' ')

    def write_yml(self, keys: list, pths: list):
        """
        Custom PyAnnote Protocol writer function that writes the required structure into a .yml file.

        Args:
            keys (list): Lower case names of train and development as per the required standard for PyAnnote training.
            pths (list): Paths to PyAnnote database directories in the order of URI, UEM and RTTM.
        """
        self.copy_audio_files([self.TRAIN, self.DEV], pb(self.WORK, "TrainDev"))
        nested_dict = lambda: defaultdict(nested_dict)
        nest = nested_dict()
        nest["Databases"]["CustomDataBase"] = pb("TrainDev", "{uri}.wav")
        for key in keys:
            for name, ext, pth in zip(["annotation", "annotated"], [".rttm", ".uem"], pths[1:]):
                nest["Protocols"]["CustomDataBase"]["SpeakerDiarization"]["default"][key][name] = pb(pb(pth, key), "{uri}" + f"{ext}")
            nest["Protocols"]["CustomDataBase"]["SpeakerDiarization"]["default"][key]["uri"] = pb(pths[0], key + f".txt")
        with open(pb(self.WORK, "database.yml"), "w") as file:
            yaml.dump(json.loads(json.dumps(nest)), file)

    def train(self):
        """
        Main training function that sets parameters, loads the model, handles logging, performs training,
        fine-tuning, and saves the model and configuration.
        """
        self.sincnet = {'stride': 10}
        self.train_epochs = 1
        self.tune_epochs = 1
        self.pretrained_pth = self.BASE_MOD_PTH
        self.batch_size = 1
        rc(self, self.TRAIN_CONF_PTH)

        model = self.load_model()
        callbacks = self.logging(model)
        trained_model_pth = self.training(model, callbacks)
        hyper_params = self.fine_tuning(trained_model_pth)
        self.save_model_and_config(trained_model_pth, hyper_params, self.OUT_PTH)

    def load_model(self):
        """
        Load the base model weights and set up the PyAnnote protocol for training.

        Returns:
            Model: Loaded model with task assigned.
        """
        registry.load_database(pb(self.WORK,"database.yml"))
        self.protocol = registry.get_protocol("CustomDataBase.SpeakerDiarization.default", preprocessors={"audio": FileFinder()})
        task = VoiceActivityDetection(self.protocol, batch_size=self.batch_size)
        
        if self.pretrained_pth is not None:
            print("Pretrained model path has been passed.")
            model = Model.from_pretrained(self.pretrained_pth)
        else:
            model = PyanNet(task=task, sincnet=self.sincnet)
        model.task = task
        return model

    def logging(self, model):
        """
        Set up logging callbacks for training, including checkpoints and early stopping.

        Args:
            model (Model): The model being trained.

        Returns:
            list: List of callbacks for training.
        """
        monitor, direction = model.task.val_monitor
        checkpoint = ModelCheckpoint(
            monitor=monitor,
            mode=direction,
            save_top_k=1,
            every_n_epochs=1,
            save_last=False,
            save_weights_only=False,
            filename="{epoch}",
            verbose=False,
        )
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode=direction,
            min_delta=0.0,
            patience=10,
            strict=True,
            verbose=False,
        )

        callbacks = [RichProgressBar(), checkpoint, early_stopping]
        return callbacks

    def training(self, model, callbacks):
        """
        Perform the training process for the model.

        Args:
            model (Model): The model being trained.
            callbacks (list): List of callbacks for training.

        Returns:
            str: Path to the best model checkpoint.
        """
        trainer = pl.Trainer(devices=1, max_epochs=self.train_epochs, callbacks=callbacks)
        trainer.fit(model.cuda())
        return callbacks[1].best_model_path
    
    def fine_tuning(self, model):
        """
        Fine-tune the trained model using the PyAnnote pipeline and optimizer.

        Args:
            model (Model): The trained model to be fine-tuned.

        Returns:
            dict: Best hyperparameters found during fine-tuning.
        """
        pipeline = VoiceActivityDetectionPipeline(segmentation=model)
        initial_params = {"onset": 0.6, "offset": 0.4,
                          "min_duration_on": 0.0, "min_duration_off": 0.0}
        pipeline.instantiate(initial_params)
        pipeline.freeze({'min_duration_on': 0.0, 'min_duration_off': 0.0})
        optimizer = Optimizer(pipeline)
        optimizer.tune(list(self.protocol.development()),
                       warm_start=initial_params,
                       n_iterations=self.tune_epochs,
                       show_progress=True)
        return optimizer.best_params

    def copy_audio_files(self, src_dirs: list, dst_dir: str):
        """
        Copy files from the input directory to the PyAnnote database directory to fulfill PyAnnote Protocol requirements.

        Args:
            src_dirs (list): List of source directories containing audio files.
            dst_dir (str): Destination directory for the copied audio files.
        """
        # Ensure the destination directory exists
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        
        for src_dir in src_dirs:
            for file_name in os.listdir(src_dir):
                if file_name.endswith('.wav'):
                    src_path = pb(src_dir, file_name)
                    dst_path = pb(dst_dir, file_name)
                    
                    # Check if the file already exists in the destination directory
                    if os.path.exists(dst_path):
                        base, ext = os.path.splitext(file_name)
                        count = 1
                        new_dst_path = pb(dst_dir, f"{base}_{count}{ext}")
                        while os.path.exists(new_dst_path):
                            count += 1
                            new_dst_path = pb(dst_dir, f"{base}_{count}{ext}")
                        dst_path = new_dst_path
                    
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
        
        print(f"Audio files have been copied to {dst_dir}")

    def save_model_and_config(self, trained_model_pth: str, hyper_params: dict, save_dir: str):
        """
        Save the trained model checkpoint and hyperparameters to the specified directory.

        Args:
            trained_model_pth (str): Path to the trained model checkpoint.
            hyper_params (dict): Best hyperparameters found during fine-tuning.
            save_dir (str): Directory to save the model and configuration.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Copy the trained model checkpoint
        ckpt_src_path = trained_model_pth
        ckpt_dst_path = pb(save_dir, 'pyannote_trained.ckpt')
        shutil.copy2(ckpt_src_path, ckpt_dst_path)
        
        # Save the hyperparameters to a YAML file with the required format
        config_path = pb(save_dir, 'pyannote_trained_config.yml')
        with open(config_path, 'w') as file:
            file.write(f"pyannote_conf: {hyper_params}")
        
        print(f"\nModel checkpoint copied to {ckpt_dst_path}")
        print(f"Hyperparameters saved to {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", help="Path to a directory containing Audio files.", required=True)
    parser.add_argument("-GND_PTH", help="Path to a directory containing GROUND TRUTH timestamps.", required=True)
    parser.add_argument("-BASE_MOD_PTH", help="Path to the base model to be trained.", required=True)
    parser.add_argument("-TRAIN_CONF_PTH", help="Path to the base config to be trained.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    args = parser.parse_args()
    Trainer(args.AUD_PTH, args.GND_PTH, args.BASE_MOD_PTH, args.TRAIN_CONF_PTH, args.OUT_PTH)
    return

if __name__ == "__main__":
    main()