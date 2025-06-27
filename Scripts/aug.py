import os
import numpy as np
import torch
import torchaudio
import yaml
import csv
import math
import librosa
from tqdm import tqdm
import random

class AudioAugmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
        # Load noise configuration
        try:
            noise_config = self.config['TimeDomain']['AddNoise']
            self.noise_csv = noise_config['csv_file']
            self.snr_low = noise_config.get('snr_low', 0)
            self.snr_high = noise_config.get('snr_high', 10)
            
            # Explicitly get noise method from config
            if 'noise_method' not in noise_config:
                raise ValueError("noise_method not specified in config")
            self.noise_method = noise_config['noise_method']
            
            # Display selected noise method
            print(f"\nNoise Augmentation Configuration:")
            print(f"Method: {self.noise_method}")
            print(f"SNR range: {self.snr_low} to {self.snr_high} dB")
            
            # Validate noise method
            valid_methods = ['spectral', 'crossfade', 'repeat']
            if self.noise_method not in valid_methods:
                raise ValueError(f"Invalid noise method: {self.noise_method}. Must be one of {valid_methods}")
    
        except KeyError as e:
            print(f"Error reading config: {e}")
            raise

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
        """Simple repetition-based synthesis that loops the same noise"""
        # Convert to numpy for easier manipulation
        if torch.is_tensor(noise):
            noise = noise.numpy()[0]
        if torch.is_tensor(audio):
            audio_length = audio.shape[1]
        else:
            audio_length = len(audio)
        
        # Get base noise pattern
        base_noise = noise
        
        # Repeat noise to match audio length
        if len(base_noise) < audio_length:
            repeats = int(np.ceil(audio_length / len(base_noise)))
            noise = np.tile(base_noise, repeats)[:audio_length]
        else:
            noise = base_noise[:audio_length]
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.01 * sr)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        noise[:fade_samples] *= fade_in
        noise[-fade_samples:] *= fade_out
        
        # Convert back to torch tensor
        return torch.from_numpy(noise).unsqueeze(0)

    def apply_crossfade_synthesis(self, audio, noise, sr):
        """Crossfade-based synthesis with smooth transitions"""
        # Get audio length and ensure noise matches
        audio_length = audio.shape[1]
        
        # Ensure noise has same number of channels as audio
        if noise.shape[0] != audio.shape[0]:
            noise = noise.repeat(audio.shape[0], 1)
        
        # Match noise length to audio length
        if noise.shape[1] != audio_length:
            noise = torch.nn.functional.interpolate(
                noise.unsqueeze(0),
                size=audio_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
    
        # Overlap parameters
        fade_duration = int(0.2 * sr)  # 200ms crossfade
        
        # Create segment envelope for smooth transitions
        envelope = torch.ones(audio_length)
        
        # Apply fade in
        if fade_duration > 0:
            fade_in = torch.cos(torch.linspace(math.pi, 2*math.pi, fade_duration)) * 0.5 + 0.5
            envelope[:fade_duration] = fade_in
    
        # Apply fade out
        if fade_duration > 0:
            fade_out = torch.cos(torch.linspace(0, math.pi, fade_duration)) * 0.5 + 0.5
            envelope[-fade_duration:] = fade_out
        
        # Apply envelope to noise
        shaped_noise = noise * envelope.unsqueeze(0)
    
        return shaped_noise

    def apply_spectral_synthesis(self, speech, noise, sr, snr):
        """
        Blend speech and noise in spectral domain using STFT
        
        Args:
            speech: Speech waveform (numpy array)
            noise: Noise waveform (numpy array)
            sr: Sample rate
            snr: Signal-to-noise ratio in dB
        """
        try:
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
            noise_scale = np.sqrt(speech_power / (noise_power * 10 ** (-snr / 10)))  # Note the negative SNR
            
            # Mix magnitudes while preserving speech phase
            mixed_mag = speech_mag + (noise_mag * noise_scale)
            mixed_stft = mixed_mag * np.exp(1j * speech_phase)
            
            # Convert back to time domain
            mixed = librosa.istft(mixed_stft, 
                                hop_length=hop_length,
                                win_length=win_length,
                                length=len(speech))  # Preserve original length
            
            return mixed
            
        except Exception as e:
            print(f"Error in spectral synthesis: {str(e)}")
            return speech

    def add_noise(self, audio, sr):
        """Add noise using specified synthesis method"""
        print(f"\nApplying noise using {self.noise_method.upper()} synthesis")

        try:
            audio_length = audio.shape[1] / sr
            mixed = audio.clone()
            noise_layer = torch.zeros_like(audio)
            noise_summary = []

            # For repeat synthesis, use single noise segment
            if self.noise_method == "repeat":
                # Select single random noise file
                noise_file = np.random.choice(self.noise_files)
                snr = np.random.uniform(self.snr_low, self.snr_high)
                
                try:
                    # Load and prepare noise
                    noise, noise_sr = torchaudio.load(noise_file['wav'])
                    
                    # Apply repeat synthesis to entire audio
                    shaped_noise = self.apply_repeat_synthesis(audio, noise, sr)
                    
                    # Apply SNR scaling
                    snr = -abs(snr)
                    noise_scale = 10 ** (snr / 20.0)
                    noise_scale = max(noise_scale, 0.3)
                    shaped_noise = shaped_noise * noise_scale
                    
                    # Add to noise layer
                    noise_layer = shaped_noise
                    
                    # Log noise info
                    noise_summary.append({
                        'segment': 1,
                        'noise_file': os.path.basename(noise_file['wav']),
                        'start': "0.0s",
                        'end': f"{audio_length:.1f}s",
                        'duration': f"{audio_length:.1f}s",
                        'snr': f"{snr:.1f}dB",
                        'method': self.noise_method
                    })
                    
                except Exception as e:
                    print(f"Error processing repeat synthesis: {str(e)}")
                    return audio
                
            else:
                # Original segmented processing for spectral and crossfade
                segments = get_dynamic_noise_segments(audio_length)
                mixed = audio.clone()
                noise_layer = torch.zeros_like(audio)
                noise_summary = []

                for i, (start_sec, end_sec) in enumerate(segments):
                    try:
                        # Convert time to samples
                        start_sample = int(start_sec * sr)
                        end_sample = int(end_sec * sr)
                        segment = audio[:, start_sample:end_sample]
                        segment_length = segment.shape[1]
                        
                        # Select random noise and SNR
                        noise_file = np.random.choice(self.noise_files)
                        snr = np.random.uniform(self.snr_low, self.snr_high)
                        
                        # Load and prepare noise
                        noise, noise_sr = torchaudio.load(noise_file['wav'])
                        
                        # Resample noise if needed
                        if noise_sr != sr:
                            noise = torchaudio.functional.resample(noise, noise_sr, sr)
                        
                        # Match noise length to segment length
                        if noise.shape[1] < segment_length:
                            repeats = int(np.ceil(segment_length / noise.shape[1]))
                            noise = noise.repeat(1, repeats)[:, :segment_length]
                        else:
                            noise = noise[:, :segment_length]
                        
                        # Normalize noise
                        noise = noise / (torch.max(torch.abs(noise)) + 1e-8)
                        
                        # Apply synthesis method
                        if self.noise_method == "spectral":
                            shaped_noise = torch.from_numpy(
                                self.apply_spectral_synthesis(
                                    segment.numpy()[0], 
                                    noise.numpy()[0], 
                                    sr, 
                                    snr
                                )
                            ).unsqueeze(0)
                        elif self.noise_method == "repeat":
                            shaped_noise = self.apply_repeat_synthesis(segment, noise, sr)
                        else:
                            shaped_noise = self.apply_crossfade_synthesis(segment, noise, sr)

                        # Ensure shaped noise matches segment length
                        if shaped_noise.shape[1] != segment_length:
                            shaped_noise = torch.nn.functional.interpolate(
                                shaped_noise.unsqueeze(0) if shaped_noise.dim() == 2 else shaped_noise,
                                size=segment_length,
                                mode='linear',
                                align_corners=False
                            ).squeeze(0)
                            if shaped_noise.dim() == 1:
                                shaped_noise = shaped_noise.unsqueeze(0)

                        # Add shaped noise to layer
                        noise_layer[:, start_sample:end_sample] = shaped_noise

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

                    except Exception as e:
                        print(f"Error processing segment {i+1}: {str(e)}")
                        continue

            # Mix and normalize
            mixed = mixed + noise_layer
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
    Calculate dynamic noise segments based on audio length
    - For long audio (>10 mins): 1 segment per ~5-8 minutes
    - Randomized segment lengths
    - Natural transitions
    """
    segments = []
    
    # Calculate number of segments based on length
    if audio_length <= 300:  # <= 5 minutes
        num_segments = np.random.randint(2, 4)
    else:
        # 1 segment per 5-8 minutes (randomized)
        mins_per_segment = np.random.uniform(5 * 60, 8 * 60)
        num_segments = max(3, int(np.ceil(audio_length / mins_per_segment)))
    
    # Create split points with variability
    split_points = [0.0]
    remaining_length = audio_length
    
    for i in range(num_segments - 1):
        # Calculate min and max lengths for this segment
        min_length = min(300, remaining_length * 0.15)  # At least 15% or 5 mins
        max_length = min(480, remaining_length * 0.3)   # At most 30% or 8 mins
        
        # Add some randomness to segment length
        segment_length = np.random.uniform(min_length, max_length)
        split_points.append(split_points[-1] + segment_length)
        remaining_length -= segment_length
    
    split_points.append(audio_length)
    
    # Convert to segments
    segments = [(split_points[i], split_points[i+1]) for i in range(len(split_points)-1)]
    
    return segments

def get_noise_segment(noise_file, segment_length, used_segments, total_length):
    max_start = total_length - segment_length
    attempts = 0
    while attempts < 10:
        start = random.randint(0, max_start)
        end = start + segment_length
        if not any(us <= start < ue or us < end <= ue for us, ue in used_segments):
            used_segments.append((start, end))
            return start, end
        attempts += 1
    return None  # fallback if too many overlaps

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
    
    # Verify config file exists
    if not os.path.exists(args.CONF_PTH):
        raise FileNotFoundError(f"Config file not found: {args.CONF_PTH}")
    
    print(f"Using config file: {args.CONF_PTH}")
    
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