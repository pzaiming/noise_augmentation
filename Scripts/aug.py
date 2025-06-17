import os
import numpy as np
import torch
import torchaudio
import yaml
import csv
from tqdm import tqdm

class AudioAugmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load noise configuration
        noise_config = self.config['TimeDomain']['AddNoise']
        self.noise_csv = noise_config['csv_file']
        self.snr_low = noise_config.get('snr_low', 0)  # More aggressive noise
        self.snr_high = noise_config.get('snr_high', 10)
        
        # Load noise files
        self.noise_files = []
        self.load_noise_files()

    def load_noise_files(self):
        try:
            with open(self.noise_csv, 'r') as f:
                reader = csv.DictReader(f)
                self.noise_files = list(reader)
            print(f"Loaded {len(self.noise_files)} noise files from {self.noise_csv}")
        except Exception as e:
            print(f"Error loading noise files: {e}")
            raise

    def apply_repeat_synthesis(self, audio, noise, sr):
        """Simple repetition-based synthesis"""
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)[:len(audio)]
        else:
            noise = noise[:len(audio)]
        return noise

    def apply_crossfade_synthesis(self, audio, noise, sr):
        """Crossfade-based synthesis"""
        fade_duration = int(0.1 * sr)
        output = np.zeros(len(audio))
        noise_pos = 0
        
        while noise_pos < len(audio):
            chunk_size = min(len(noise), len(audio) - noise_pos)
            noise_chunk = noise[:chunk_size]
            
            if noise_pos > 0:
                fade_in = np.linspace(0, 1, fade_duration)
                fade_out = np.linspace(1, 0, fade_duration)
                
                output[noise_pos:noise_pos+fade_duration] *= fade_out
                noise_chunk[:fade_duration] *= fade_in
            
            output[noise_pos:noise_pos+chunk_size] += noise_chunk
            noise_pos += chunk_size - fade_duration
        
        return output

    def apply_spectral_synthesis(self, audio, noise, sr):
        """Spectral domain synthesis"""
        n_fft = 2048
        noise_stft = librosa.stft(noise, n_fft=n_fft)
        noise_mag = np.abs(noise_stft)
        
        # Create extended noise in TF domain
        n_frames = int(np.ceil(len(audio) / (n_fft/4)))
        noise_extended = np.zeros((n_fft//2 + 1, n_frames), dtype=np.complex128)
        
        for f in range(noise_mag.shape[0]):
            pattern = noise_mag[f, :]
            repeats = int(np.ceil(n_frames / len(pattern)))
            extended = np.tile(pattern, repeats)[:n_frames]
            phase = np.random.uniform(-np.pi, np.pi, n_frames)
            noise_extended[f, :] = extended * np.exp(1j * phase)
        
        return librosa.istft(noise_extended, length=len(audio))

    def add_noise(self, audio, noise_file, snr):
        """Add noise with guaranteed audibility"""
        try:
            # Load and prepare noise
            noise, noise_sr = torchaudio.load(noise_file['wav'])
            
            # Print debug info
            print(f"\nNoise file: {noise_file['wav']}")
            print(f"Noise range: {torch.min(noise).item():.3f} to {torch.max(noise).item():.3f}")
            print(f"Audio range: {torch.min(audio).item():.3f} to {torch.max(audio).item():.3f}")
            
            # Match lengths
            if noise.shape[1] < audio.shape[1]:
                repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
                noise = noise.repeat(1, repeats)[:, :audio.shape[1]]
            else:
                noise = noise[:, :audio.shape[1]]

            # Normalize both signals to peak amplitude 1
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            noise = noise / (torch.max(torch.abs(noise)) + 1e-8)
            
            # Calculate noise scale based on desired SNR
            # Using negative SNR for more prominent noise
            snr = -abs(snr)  # Make SNR negative to increase noise
            noise_scale = 10 ** (snr / 20.0)  # Convert to amplitude ratio
            
            # Ensure minimum noise level
            noise_scale = max(noise_scale, 0.3)  # At least 30% noise
            
            # Mix signals
            mixed = audio + (noise * noise_scale)
            
            # Print debug info
            print(f"SNR: {snr:.1f} dB")
            print(f"Noise scale: {noise_scale:.3f}")
            print(f"Mixed range: {torch.min(mixed).item():.3f} to {torch.max(mixed).item():.3f}")
            
            # Final normalization
            mixed = mixed / (torch.max(torch.abs(mixed)) + 1e-8) * 0.95
            
            return mixed
            
        except Exception as e:
            print(f"Error adding noise: {str(e)}")
            return audio

    def process_file(self, input_path, output_path):
        try:
            # Load audio without explicit backend setting
            audio, sr = torchaudio.load(input_path)
            
            # Select random noise and SNR
            noise_file = np.random.choice(self.noise_files)
            snr = np.random.uniform(self.snr_low, self.snr_high)
            
            # Apply noise augmentation
            augmented = self.add_noise(audio, noise_file, snr)
            
            # Save augmented audio
            torchaudio.save(output_path, augmented, sr)
            return True
            
        except Exception as e:
            print(f"\nError processing {input_path}: {e}")
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