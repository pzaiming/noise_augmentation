import os
import numpy as np
import torch
import torchaudio
import yaml
import csv
import math
import librosa
from tqdm import tqdm

class AudioAugmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
        # Load noise configuration
        try:
            noise_config = self.config['TimeDomain']['AddNoise']
            self.noise_base_dir = noise_config['noise_base_dir']  # Directory containing noise clusters
            self.snr_low = noise_config.get('snr_low', 0)
            self.snr_high = noise_config.get('snr_high', 10)
            
            # Explicitly get noise method from config
            if 'noise_method' not in noise_config:
                raise ValueError("noise_method not specified in config")
            self.noise_method = noise_config['noise_method']
            
            # Get clusters
            self.noise_clusters = self.load_noise_clusters()
            
            # Display configuration
            print(f"\nNoise Augmentation Configuration:")
            print(f"Method: {self.noise_method}")
            print(f"SNR range: {self.snr_low} to {self.snr_high} dB")
            print(f"Number of noise clusters: {len(self.noise_clusters)}")
            
            # Validate noise method
            valid_methods = ['spectral', 'crossfade', 'repeat']
            if self.noise_method not in valid_methods:
                raise ValueError(f"Invalid noise method: {self.noise_method}. Must be one of {valid_methods}")
    
        except KeyError as e:
            print(f"Error reading config: {e}")
            raise

    def load_noise_clusters(self):
        """Load noise files organized in clusters"""
        clusters = []
        try:
            # List all cluster directories
            cluster_dirs = [d for d in os.listdir(self.noise_base_dir) 
                          if os.path.isdir(os.path.join(self.noise_base_dir, d)) 
                          and d.startswith('cluster')]
            
            # Sort clusters by SNR range
            cluster_dirs.sort(key=lambda x: float(x.split('snr')[1].split('to')[0]))
            
            # Load files from each cluster
            for cluster_dir in cluster_dirs:
                cluster_path = os.path.join(self.noise_base_dir, cluster_dir)
                noise_files = []
                
                # Get all audio files in the cluster
                for file in os.listdir(cluster_path):
                    if file.endswith(('.wav', '.mp3', '.flac')):
                        file_path = os.path.join(cluster_path, file)
                        noise_files.append({
                            'path': file_path,
                            'cluster': cluster_dir
                        })
                
                if noise_files:
                    # Extract SNR range from directory name
                    snr_range = cluster_dir.split('snr')[1].split('to')
                    clusters.append({
                        'name': cluster_dir,
                        'files': noise_files,
                        'snr_range': (float(snr_range[0]), float(snr_range[1]))
                    })
            
            print(f"\nLoaded {len(clusters)} noise clusters:")
            for cluster in clusters:
                print(f"  {cluster['name']}: {len(cluster['files'])} files, "
                      f"SNR range: {cluster['snr_range'][0]:.1f} to {cluster['snr_range'][1]:.1f} dB")
            
            return clusters
            
        except Exception as e:
            print(f"Error loading noise clusters: {e}")
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

    def add_noise(self, audio, sr, noise, snr_low=None, snr_high=None):
        """Add noise using specified synthesis method with optional SNR range override"""
        print(f"\nApplying noise using {self.noise_method.upper()} synthesis")
        
        # Use cluster-specific SNR range if provided
        snr_range_low = snr_low if snr_low is not None else self.snr_low
        snr_range_high = snr_high if snr_high is not None else self.snr_high

        try:
            audio_length = audio.shape[1] / sr
            mixed = audio.clone()
            noise_layer = torch.zeros_like(audio)

            if self.noise_method == "repeat":
                # Use the provided noise
                noise_segment = self.apply_repeat_synthesis(audio, noise, sr)
                noise_layer = noise_segment
                
            elif self.noise_method == "crossfade":
                noise_layer = self.apply_crossfade_synthesis(audio, noise, sr)
                
            elif self.noise_method == "spectral":
                # Convert to numpy for spectral processing
                audio_np = audio.numpy()[0]
                noise_np = noise.numpy()[0]
                
                # Apply spectral synthesis with cluster-specific SNR range
                mixed_np = self.apply_spectral_synthesis(
                    audio_np, noise_np, sr, 
                    np.random.uniform(snr_range_low, snr_range_high)
                )
                
                # Convert back to torch tensor
                mixed = torch.from_numpy(mixed_np).unsqueeze(0)
                return mixed

            # For non-spectral methods, mix and normalize
            if self.noise_method != "spectral":
                mixed = mixed + noise_layer
                mixed = mixed / (torch.max(torch.abs(mixed)) + 1e-8) * 0.95

            return mixed

        except Exception as e:
            print(f"Error adding noise: {str(e)}")
            return audio

    def process_file(self, input_path, output_path):
        """
        Process a single audio file and create multiple augmented versions,
        one for each noise cluster
        """
        try:
            # Load audio file
            audio, sr = torchaudio.load(input_path)
            base_name, ext = os.path.splitext(output_path)
            
            augmented_files = []
            
            # Create one augmented version per cluster
            for cluster in self.noise_clusters:
                # Select random noise file from this cluster
                noise_file = np.random.choice(cluster['files'])
                noise_path = noise_file['path']
                
                # Load and prepare noise
                noise, noise_sr = torchaudio.load(noise_path)
                if noise_sr != sr:
                    noise = torchaudio.functional.resample(noise, noise_sr, sr)
                
                # Apply noise using selected method with cluster's SNR range
                snr_range_low, snr_range_high = cluster['snr_range']
                augmented = self.add_noise(audio, sr, noise, snr_range_low, snr_range_high)
                
                # Create output filename with cluster info
                cluster_output = f"{base_name}_cluster{cluster['name'].split('cluster')[1]}{ext}"
                
                # Get dynamic segments based on audio length
                audio_duration = audio.shape[1] / sr
                segments = get_dynamic_noise_segments(audio_duration)
                
                # Initialize noise summary
                noise_summary = []
                
                # Process each segment
                for idx, (start_time, end_time) in enumerate(segments, 1):
                    # Select random noise file from this cluster for each segment
                    noise_file = np.random.choice(cluster['files'])
                    segment_snr = np.random.uniform(snr_range_low, snr_range_high)
                    
                    noise_summary.append({
                        'segment': idx,
                        'noise_file': os.path.basename(noise_file['path']),
                        'duration': f"{end_time - start_time:.2f}s",
                        'start': f"{start_time:.2f}s",
                        'end': f"{end_time:.2f}s",
                        'snr': f"{segment_snr:.2f} dB",
                        'method': self.noise_method
                    })
                
                # Save augmented audio
                torchaudio.save(cluster_output, augmented, sr)
                augmented_files.append(cluster_output)
                
                print(f"\nCreated augmented version: {os.path.basename(cluster_output)}")
                print("Noise segments used:")
                for segment in noise_summary:
                    print(f"Segment {segment['segment']}: {segment['noise_file']}")
                    print(f"  Duration: {segment['duration']} ({segment['start']} - {segment['end']})")
                    print(f"  SNR: {segment['snr']}")
                    print(f"  Method: {segment['method']}")
                print("-" * 50)
                
            return augmented_files
                
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            return None

def get_dynamic_noise_segments(audio_length):
    """
    For repeat synthesis, use one long segment
    """
    if audio_length <= 5.0:
        segments = [(0.0, audio_length)]
    else:
        # For longer audio, use 2-3 segments
        num_segments = np.random.randint(2, 4)
        split_points = [0.0]
        random_splits = sorted(np.random.uniform(0.2, 0.8, num_segments-1))
        split_points.extend(random_splits)
        split_points.append(1.0)
        
        segments = []
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