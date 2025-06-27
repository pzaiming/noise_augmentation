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

    def format_time(self, seconds):
        """Convert seconds to minutes and seconds format"""
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        if minutes > 0:
            return f"{minutes}m {remaining_seconds:.1f}s"
        return f"{remaining_seconds:.1f}s"

    def add_noise(self, audio, sr):
        """Add multiple noise segments using specified synthesis method"""
        try:
            audio_length = audio.shape[1] / sr
            print(f"\nAudio length: {self.format_time(audio_length)}")
            print(f"\nApplying noise using {self.noise_method.upper()} synthesis")

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
                    
                    # Select random noise and SNR
                    noise_file = np.random.choice(self.noise_files)
                    snr = np.random.uniform(self.snr_low, self.snr_high)
                    
                    # Load and prepare noise
                    noise, noise_sr = torchaudio.load(noise_file['wav'])
                    
                    # Resample noise if needed
                    if noise_sr != sr:
                        noise = torchaudio.functional.resample(noise, noise_sr, sr)
                    
                    # Match noise length to segment length
                    segment_length = segment.shape[1]
                    if noise.shape[1] != segment_length:
                        noise = torch.nn.functional.interpolate(
                            noise.unsqueeze(0),
                            size=segment_length,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0)

                    # Apply synthesis based on method
                    if self.noise_method == "spectral":
                        shaped_noise = self.apply_spectral_synthesis(
                            segment.numpy()[0],
                            noise.numpy()[0],
                            sr,
                            snr
                        )
                        shaped_noise = torch.from_numpy(shaped_noise).unsqueeze(0)
                    elif self.noise_method == "repeat":
                        shaped_noise = self.apply_repeat_synthesis(segment, noise, sr)
                        # Apply SNR scaling
                        snr_factor = 10 ** (-snr / 20)
                        shaped_noise = shaped_noise * snr_factor
                    else:  # crossfade
                        shaped_noise = self.apply_crossfade_synthesis(segment, noise, sr)
                        # Apply SNR scaling
                        snr_factor = 10 ** (-snr / 20)
                        shaped_noise = shaped_noise * snr_factor

                    # Add shaped noise to layer
                    noise_layer[:, start_sample:end_sample] = shaped_noise

                    # Add debug info
                    print(f"\nDebug - Segment {i+1}:")
                    print(f"Max speech amplitude: {torch.max(torch.abs(segment)):.6f}")
                    print(f"Max noise amplitude: {torch.max(torch.abs(shaped_noise)):.6f}")
                    print(f"SNR factor: {snr_factor if self.noise_method != 'spectral' else 'N/A'}")

                    # Store segment info
                    noise_summary.append({
                        'segment': i + 1,
                        'noise_file': os.path.basename(noise_file['wav']),
                        'start': start_sec,
                        'end': end_sec,
                        'duration': end_sec - start_sec,
                        'snr': snr,
                        'method': self.noise_method
                    })

                except Exception as e:
                    print(f"Error processing segment {i+1}: {str(e)}")
                    continue

            # Mix audio with noise
            mixed = mixed + noise_layer * 0.7  # Adjust noise level

            # Normalize final output
            max_val = torch.max(torch.abs(mixed))
            mixed = mixed / max_val * 0.95

            # Print summary
            print("\nNoise segments used:")
            for seg in noise_summary:
                print(f"Segment {seg['segment']}: {seg['noise_file']}")
                print(f"  Duration: {self.format_time(seg['duration'])} "
                    f"({self.format_time(seg['start'])} - {self.format_time(seg['end'])})")
                print(f"  SNR: {seg['snr']:.1f}dB")
                print(f"  Method: {seg['method']}")

            return mixed

        except Exception as e:
            print(f"Error adding noise: {str(e)}")
            import traceback
            traceback.print_exc()
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
    Systematic approach for noise segmentation:
    
    1. Short clips (< 1 min):
       - Single noise segment
    
    2. Medium clips (1-5 mins):
       - 2 segments with golden ratio split
       - Prevents repetition
    
    3. Long clips (5-15 mins):
       - 3-4 segments
       - Min segment length: 2 mins
       - Max segment length: 5 mins
    
    4. Extended clips (> 15 mins):
       - 1 segment per 4-6 mins
       - Random variation in lengths
       - No consecutive repetition
    """
    segments = []
    
    # Constants for segment calculation
    GOLDEN_RATIO = 0.618
    MIN_SEGMENT = 120  # 2 mins minimum
    MAX_SEGMENT = 300  # 5 mins maximum
    
    if audio_length < 60:  # Short clips
        segments = [(0.0, audio_length)]
        
    elif audio_length <= 300:  # Medium clips (1-5 mins)
        # Use golden ratio for natural-feeling split
        split_point = audio_length * GOLDEN_RATIO
        segments = [
            (0.0, split_point),
            (split_point, audio_length)
        ]
        
    elif audio_length <= 900:  # Long clips (5-15 mins)
        num_segments = np.random.randint(3, 5)
        remaining_length = audio_length
        current_point = 0.0
        
        while remaining_length > 0 and len(segments) < num_segments:
            # Calculate segment length
            min_len = min(MIN_SEGMENT, remaining_length * 0.2)
            max_len = min(MAX_SEGMENT, remaining_length * 0.4)
            segment_length = np.random.uniform(min_len, max_len)
            
            segments.append((current_point, current_point + segment_length))
            current_point += segment_length
            remaining_length -= segment_length
            
        # Add final segment if needed
        if remaining_length > 0:
            segments.append((current_point, audio_length))
            
    else:  # Extended clips (> 15 mins)
        # Calculate number of segments (1 per 4-6 minutes)
        segment_duration = np.random.uniform(240, 360)
        num_segments = max(4, int(np.ceil(audio_length / segment_duration)))
        remaining_length = audio_length
        current_point = 0.0
        
        for i in range(num_segments - 1):
            # Vary segment length by ±20%
            base_length = remaining_length / (num_segments - i)
            variation = base_length * 0.2
            segment_length = np.random.uniform(
                base_length - variation,
                base_length + variation
            )
            
            segments.append((current_point, current_point + segment_length))
            current_point += segment_length
            remaining_length -= segment_length
        
        # Add final segment
        segments.append((current_point, audio_length))
    
    return segments

def select_noise_files(self, num_segments):
    """
    Select noise files ensuring no consecutive repetition
    """
    selected_files = []
    available_files = self.noise_files.copy()
    
    for i in range(num_segments):
        if not available_files:
            # Reset pool but exclude recently used files
            available_files = [n for n in self.noise_files 
                             if n not in selected_files[-2:]]
        
        # Select random file
        noise_file = np.random.choice(available_files)
        available_files.remove(noise_file)
        selected_files.append(noise_file)
    
    return selected_files

def main():
    import argparse
    from tqdm import tqdm
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", required=True)
    parser.add_argument("-CONF_PTH", required=True)
    parser.add_argument("-OUT_PTH", required=True)
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.OUT_PTH, exist_ok=True)
    
    # Initialize augmenter
    print("Initializing augmenter...")
    augmenter = AudioAugmenter(args.CONF_PTH)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(args.AUD_PTH) if f.endswith('.mp3') or f.endswith('.wav')]
    
    # Process each file with progress bar
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        print("\nProcessing:", audio_file)
        print("-" * 50)
        
        input_path = os.path.join(args.AUD_PTH, audio_file)
        output_path = os.path.join(args.OUT_PTH, f"aug_{audio_file}")
        
        try:
            # Load audio
            audio, sr = torchaudio.load(input_path)
            
            # Apply augmentation
            augmented = augmenter.add_noise(audio, sr)
            
            # Save augmented audio
            torchaudio.save(output_path, augmented, sr)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()