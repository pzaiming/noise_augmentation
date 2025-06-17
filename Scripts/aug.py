import os
import numpy as np
import torch
import torchaudio
import yaml
import csv
import math
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

    def add_noise(self, audio, sr):
        """Add multiple noise segments with different profiles"""
        try:
            # Convert audio length from samples to seconds
            audio_length = audio.shape[1] / sr
            
            # Get dynamic segments for this audio length
            segments = get_dynamic_noise_segments(audio_length)
            
            # Initialize output tensor
            mixed = audio.clone()
            
            # Apply different noise to each segment
            for start_sec, end_sec in segments:
                # Convert time to samples
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                
                # Select random noise file and SNR for this segment
                noise_file = np.random.choice(self.noise_files)
                snr = np.random.uniform(self.snr_low, self.snr_high)
                
                # Load and prepare noise
                noise, noise_sr = torchaudio.load(noise_file['wav'])
                
                # Extract audio segment
                segment = audio[:, start_sample:end_sample]
                
                # Match noise length to segment length
                if noise.shape[1] < segment.shape[1]:
                    repeats = int(np.ceil(segment.shape[1] / noise.shape[1]))
                    noise = noise.repeat(1, repeats)[:, :segment.shape[1]]
                else:
                    noise = noise[:, :segment.shape[1]]

                # Normalize both signals
                segment = segment / (torch.max(torch.abs(segment)) + 1e-8)
                noise = noise / (torch.max(torch.abs(noise)) + 1e-8)
                
                # Calculate noise scale
                snr = -abs(snr)  # Make SNR negative for more prominent noise
                noise_scale = 10 ** (snr / 20.0)
                noise_scale = max(noise_scale, 0.3)  # Minimum noise level
                
                # Apply noise to segment
                mixed_segment = segment + (noise * noise_scale)
                
                # Normalize segment
                mixed_segment = mixed_segment / (torch.max(torch.abs(mixed_segment)) + 1e-8)
                
                # Apply crossfade at segment boundaries
                fade_samples = int(0.05 * sr)  # 50ms crossfade
                if start_sample > 0:
                    # Fade in
                    fade_in = torch.linspace(0, 1, fade_samples)
                    mixed_segment[:, :fade_samples] *= fade_in
                    mixed[:, start_sample:start_sample+fade_samples] *= (1 - fade_in)
                
                if end_sample < audio.shape[1]:
                    # Fade out
                    fade_out = torch.linspace(1, 0, fade_samples)
                    mixed_segment[:, -fade_samples:] *= fade_out
                    mixed[:, end_sample-fade_samples:end_sample] *= (1 - fade_out)
                
                # Insert mixed segment into output
                mixed[:, start_sample:end_sample] = mixed_segment
                
                print(f"Applied noise from {noise_file['wav']} at {start_sec:.1f}-{end_sec:.1f}s with SNR {snr:.1f}dB")
            
            # Final normalization
            mixed = mixed / (torch.max(torch.abs(mixed)) + 1e-8) * 0.95
            
            return mixed
            
        except Exception as e:
            print(f"Error adding noise: {str(e)}")
            return audio

    def process_file(self, input_path, output_path):
        try:
            # Load audio
            audio, sr = torchaudio.load(input_path)
            
            # Apply noise augmentation with multiple profiles
            augmented = self.add_noise(audio, sr)
            
            # Save augmented audio
            torchaudio.save(output_path, augmented, sr)
            return True
            
        except Exception as e:
            print(f"\nError processing {input_path}: {e}")
            return False

def get_dynamic_noise_segments(audio_length, min_noise_len=1.0, max_noise_len=2.0):
    """
    Calculate dynamic noise segment positions based on audio length
    
    Parameters:
    - audio_length: Length of the audio in seconds
    - min_noise_len: Minimum length of a noise segment (default 1.0s)
    - max_noise_len: Maximum length of a noise segment (default 2.0s)
    
    Returns: List of (start_time, end_time) tuples
    """
    segments = []
    
    # Calculate number of segments based on audio length
    # Longer audio gets more segments
    num_segments = max(2, math.ceil(audio_length / 3))
    
    # Dynamic noise length based on audio length
    # Longer audio gets longer noise segments, up to max_noise_len
    noise_length = min(max_noise_len, 
                      max(min_noise_len, audio_length / (num_segments * 1.5)))
    
    # Calculate spacing between segments
    available_space = audio_length - (noise_length * num_segments)
    gap = available_space / (num_segments + 1)
    
    # Create segments with even spacing
    for i in range(num_segments):
        start = (gap * (i + 1)) + (noise_length * i)
        end = start + noise_length
        segments.append((start, end))
    
    return segments

# Example usage:
examples = [3.0, 4.5, 6.0, 8.0]
for length in examples:
    segments = get_dynamic_noise_segments(length)
    print(f"\n{length}s audio segments:")
    print(f"Number of segments: {len(segments)}")
    print(f"Segments: {segments}")

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