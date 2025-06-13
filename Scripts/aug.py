import os
import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

class AudioAugmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load noise files
        noise_csv = self.config['TimeDomain']['AddNoise']['csv_file']
        self.load_noise_files(noise_csv)
    
    def load_noise_files(self, csv_path):
        import csv
        self.noise_files = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self.noise_files = list(reader)
    
    def add_noise(self, audio_path, noise_path, snr):
        """Simple noise addition that preserves original audio"""
        # Load original audio
        audio, sr = torchaudio.load(audio_path)
        
        # Load and prepare noise
        noise, noise_sr = torchaudio.load(noise_path)
        
        # Resample noise if needed
        if noise_sr != sr:
            noise = torchaudio.transforms.Resample(noise_sr, sr)(noise)
        
        # Convert both to mono if needed
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        
        # Adjust noise length
        if noise.shape[1] < audio.shape[1]:
            # Repeat noise to match audio length
            repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
            noise = noise.repeat(1, repeats)[:, :audio.shape[1]]
        else:
            # Trim noise to match audio length
            noise = noise[:, :audio.shape[1]]
        
        # Calculate mixing ratio based on SNR
        audio_power = audio.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        # Convert SNR to scaling factor
        scaling = torch.sqrt(audio_power / (noise_power * (10 ** (snr/10))))
        
        # Mix audio with noise
        noisy_audio = audio + (noise * scaling)
        
        # Normalize if needed
        max_val = torch.abs(noisy_audio).max()
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val * 0.9
        
        return noisy_audio, sr
    
    def process_file(self, input_path, output_path):
        try:
            # Select random noise file
            noise_file = np.random.choice(self.noise_files)
            noise_path = noise_file['wav']
            
            # Get SNR from config
            noise_config = self.config['TimeDomain']['AddNoise']
            snr = np.random.uniform(noise_config['snr_low'], noise_config['snr_high'])
            
            # Add noise
            augmented, sr = self.add_noise(input_path, noise_path, snr)
            
            # Save with same parameters as input
            torchaudio.save(output_path, augmented, sr)
            return True
        
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", required=True)
    parser.add_argument("-CONF_PTH", required=True)
    parser.add_argument("-OUT_PTH", required=True)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.OUT_PTH, exist_ok=True)
    
    # Initialize augmenter
    augmenter = AudioAugmenter(args.CONF_PTH)
    
    # Get audio files
    audio_files = [f for f in os.listdir(args.AUD_PTH) 
                   if f.endswith(('.wav', '.mp3', '.flac'))]
    
    # Process files
    for file in tqdm(audio_files):
        input_path = os.path.join(args.AUD_PTH, file)
        output_path = os.path.join(args.OUT_PTH, f"aug_{file}")
        augmenter.process_file(input_path, output_path)

if __name__ == "__main__":
    main()