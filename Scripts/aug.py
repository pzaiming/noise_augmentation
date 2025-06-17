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
        self.snr_low = noise_config.get('snr_low', 0)
        self.snr_high = noise_config.get('snr_high', 10)
        self.noise_method = noise_config.get('noise_method', 'crossfade')  # Default to crossfade
    
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
        # Convert to numpy for easier manipulation
        if torch.is_tensor(noise):
            noise = noise.numpy()[0]
        if torch.is_tensor(audio):
            audio_length = audio.shape[1]
        else:
            audio_length = len(audio)
        
        # Repeat noise to match audio length
        if len(noise) < audio_length:
            repeats = int(np.ceil(audio_length / len(noise)))
            noise = np.tile(noise, repeats)[:audio_length]
        else:
            noise = noise[:audio_length]
        
        # Convert back to torch tensor
        return torch.from_numpy(noise).unsqueeze(0)

    def apply_crossfade_synthesis(self, audio, noise, sr):
        """Crossfade-based synthesis with smooth transitions"""
        # Get audio length and ensure noise matches
        audio_length = audio.shape[1]
        if noise.shape[1] != audio_length:
            if noise.shape[1] < audio_length:
                noise = noise.repeat(1, int(np.ceil(audio_length / noise.shape[1])))
            noise = noise[:, :audio_length]
    
        # Overlap parameters
        fade_duration = int(0.2 * sr)  # 200ms crossfade
        
        # Create segment envelope for smooth transitions
        envelope = torch.ones(audio_length)
        
        # Apply fade in
        fade_in = torch.cos(torch.linspace(math.pi, 2*math.pi, fade_duration)) * 0.5 + 0.5
        envelope[:fade_duration] = fade_in
        
        # Apply fade out
        fade_out = torch.cos(torch.linspace(0, math.pi, fade_duration)) * 0.5 + 0.5
        envelope[-fade_duration:] = fade_out
        
        # Apply envelope to noise
        shaped_noise = noise * envelope.unsqueeze(0)
    
        return shaped_noise

    def apply_spectral_synthesis(self, speech, noise, sr, snr):
        """
        Blend speech and noise in spectral domain
        
        Args:
            speech: Speech waveform
            noise: Noise waveform
            sr: Sample rate
            snr: Signal-to-noise ratio in dB
        """
        # STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
        win_length = n_fft
        
        # Convert to spectrograms
        speech_stft = librosa.stft(speech, n_fft=n_fft, 
                                  hop_length=hop_length,
                                  win_length=win_length)
        noise_stft = librosa.stft(noise, n_fft=n_fft,
                                 hop_length=hop_length,
                                 win_length=win_length)
        
        # Get magnitudes and phases
        speech_mag = np.abs(speech_stft)
        speech_phase = np.angle(speech_stft)
        noise_mag = np.abs(noise_stft)
        noise_phase = np.angle(noise_stft)
        
        # Calculate noise scale based on SNR
        speech_power = np.mean(speech_mag ** 2)
        noise_power = np.mean(noise_mag ** 2)
        noise_scale = np.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))
        
        # Blend magnitudes
        mixed_mag = speech_mag + (noise_mag * noise_scale)
        
        # Combine with original speech phase
        mixed_stft = mixed_mag * np.exp(1j * speech_phase)
        
        # Convert back to time domain
        mixed = librosa.istft(mixed_stft, hop_length=hop_length, win_length=win_length)
        
        return mixed

    def add_noise(self, audio, sr):
        """Add multiple noise segments using specified synthesis method"""
        try:
            audio_length = audio.shape[1] / sr
            segments = get_dynamic_noise_segments(audio_length)
            mixed = audio.clone()
            noise_layer = torch.zeros_like(audio)
            noise_summary = []
            
            for i, (start_sec, end_sec) in enumerate(segments):
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                
                # Select random noise file and SNR
                noise_file = np.random.choice(self.noise_files)
                snr = np.random.uniform(self.snr_low, self.snr_high)
                
                # Load and prepare noise
                noise, noise_sr = torchaudio.load(noise_file['wav'])
                segment_length = end_sample - start_sample
                
                # Match noise length to segment
                if noise.shape[1] < segment_length:
                    repeats = int(np.ceil(segment_length / noise.shape[1]))
                    noise = noise.repeat(1, repeats)[:, :segment_length]
                else:
                    noise = noise[:, :segment_length]
                
                # Normalize noise
                noise = noise / (torch.max(torch.abs(noise)) + 1e-8)
                
                # Apply selected synthesis method
                if self.noise_method == "repeat":
                    shaped_noise = self.apply_repeat_synthesis(audio[:, start_sample:end_sample], 
                                                             noise, sr)
                elif self.noise_method == "spectral":
                    shaped_noise = self.apply_spectral_synthesis(audio[:, start_sample:end_sample].numpy()[0], 
                                                               noise.numpy()[0], sr, snr)
                    shaped_noise = torch.from_numpy(shaped_noise).unsqueeze(0)
                else:  # default to crossfade
                    shaped_noise = self.apply_crossfade_synthesis(audio[:, start_sample:end_sample], 
                                                                noise, sr)
                
                # Calculate noise scale (except for spectral which handles SNR internally)
                if self.noise_method != "spectral":
                    snr = -abs(snr)
                    noise_scale = 10 ** (snr / 20.0)
                    noise_scale = max(noise_scale, 0.3)
                    shaped_noise = shaped_noise * noise_scale
                
                # Add shaped noise to noise layer
                noise_layer[:, start_sample:end_sample] += shaped_noise
            
                # Store noise info
                noise_summary.append({
                    'segment': i + 1,
                    'noise_file': os.path.basename(noise_file['wav']),
                    'start': f"{start_sec:.1f}s",
                    'end': f"{end_sec:.1f}s",
                    'duration': f"{end_sec - start_sec:.1f}s",
                    'snr': f"{snr:.1f}dB",
                    'method': self.noise_method
                })
            
            # Normalize and mix
            noise_layer = noise_layer / (torch.max(torch.abs(noise_layer)) + 1e-8)
            mixed = mixed + (noise_layer * 0.7)
            mixed = mixed / (torch.max(torch.abs(mixed)) + 1e-8) * 0.95
            
            # Print summary
            print("\nNoise segments used:")
            for segment in noise_summary:
                print(f"Segment {segment['segment']}: {segment['noise_file']}")
                print(f"  Duration: {segment['duration']} ({segment['start']} - {segment['end']})")
                print(f"  SNR: {segment['snr']}")
                print(f"  Method: {segment['method']}")
        
            return mixed
        
        except Exception as e:
            print(f"Error adding noise: {str(e)}")
            return audio

    def process_file(self, input_path, output_path):
        try:
            print(f"\nProcessing: {os.path.basename(input_path)}")
            print("-" * 50)
            
            # Load audio
            audio, sr = torchaudio.load(input_path)
            audio_length = audio.shape[1] / sr
            print(f"Audio length: {audio_length:.2f}s")
            
            # Apply noise augmentation with multiple profiles
            augmented = self.add_noise(audio, sr)
            
            # Save augmented audio
            torchaudio.save(output_path, augmented, sr)
            print("-" * 50)
            return True
            
        except Exception as e:
            print(f"\nError processing {input_path}: {e}")
            return False

def get_dynamic_noise_segments(audio_length):
    """
    Calculate noise segments based on audio length:
    - For clips <= 5s: single noise
    - For clips > 5s: 2-3 segments
    
    Parameters:
    - audio_length: Length of the audio in seconds
    
    Returns: List of (start_time, end_time) tuples
    """
    segments = []
    
    # For very short clips (<= 5s), use single noise
    if audio_length <= 5.0:
        segments.append((0.0, audio_length))
        return segments
    
    # For longer clips, use 2-3 segments
    num_segments = np.random.randint(2, 4)  # 2 or 3 segments
    
    # Create random split points
    split_points = [0.0]  # Start point
    random_splits = sorted(np.random.uniform(0.2, 0.8, num_segments-1))
    split_points.extend(random_splits)
    split_points.append(1.0)  # End point
    
    # Convert split points to actual time segments
    for i in range(len(split_points)-1):
        start = split_points[i] * audio_length
        end = split_points[i+1] * audio_length
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