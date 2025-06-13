import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import csv
from scipy import signal
from tqdm import tqdm

class BackgroundNoiseExtractor:
    def __init__(self, audio_path, output_path, frame_size=0.1, 
                 noise_percentile=20, output_duration=2.0):
        self.audio_path = audio_path
        self.output_path = output_path
        self.noise_dir = os.path.join(output_path, "Noise")
        os.makedirs(self.noise_dir, exist_ok=True)
        
        self.frame_size = frame_size
        self.noise_percentile = noise_percentile
        self.output_duration = output_duration
        self.noise_files = []

    def extract_background_profile(self, audio_data, sr):
        """Extract persistent background noise profile"""
        # Compute STFT
        n_fft = int(self.frame_size * sr)
        hop_length = n_fft // 4
        
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        
        # For each frequency bin, find the consistent background level
        # using the specified percentile (ignoring silence)
        background_profile = np.zeros(magnitude.shape[0])
        
        for freq_idx in range(magnitude.shape[0]):
            # Get magnitudes for this frequency
            freq_mags = magnitude[freq_idx]
            # Remove near-zero values (silence)
            non_silent = freq_mags[freq_mags > np.max(freq_mags) * 0.01]
            if len(non_silent) > 0:
                # Use percentile to find consistent background level
                background_profile[freq_idx] = np.percentile(
                    non_silent, self.noise_percentile)
        
        return background_profile, n_fft, hop_length

    def synthesize_background_noise(self, profile, n_fft, hop_length, duration, sr):
        """Synthesize background noise from profile"""
        n_frames = int((duration * sr) / hop_length) + 1
        
        # Create noise spectrogram
        noise_spec = np.zeros((len(profile), n_frames), dtype=np.complex128)
        
        for frame in range(n_frames):
            # Use profile magnitudes with random phase
            phase = np.random.uniform(-np.pi, np.pi, size=len(profile))
            noise_spec[:, frame] = profile * np.exp(1j * phase)
        
        # Convert back to time domain
        noise = librosa.istft(noise_spec, hop_length=hop_length, win_length=n_fft)
        
        # Trim to exact duration
        target_length = int(duration * sr)
        if len(noise) > target_length:
            noise = noise[:target_length]
        else:
            noise = np.pad(noise, (0, target_length - len(noise)))
        
        return noise

    def extract_noise_from_file(self, audio_file):
        """Extract background noise from a single file"""
        print(f"Processing {audio_file}...")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Get background noise profile
        profile, n_fft, hop_length = self.extract_background_profile(y, sr)
        
        # Create multiple noise samples
        extracted_noise = []
        num_samples = 3  # Number of noise files to generate
        
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        
        for i in range(num_samples):
            # Synthesize noise
            noise = self.synthesize_background_noise(
                profile, n_fft, hop_length, self.output_duration, sr)
            
            # Save to file
            noise_filename = f"{basename}_background_{i+1}.wav"
            noise_path = os.path.join(self.noise_dir, noise_filename)
            sf.write(noise_path, noise, sr)
            
            noise_info = {
                'path': noise_path,
                'filename': noise_filename,
                'duration': self.output_duration,
                'original_file': os.path.basename(audio_file)
            }
            extracted_noise.append(noise_info)
            print(f"Created background noise file {i+1} from {basename}")
        
        return extracted_noise

    def process(self):
        """Process all audio files"""
        if os.path.isdir(self.audio_path):
            audio_files = [
                os.path.join(self.audio_path, f) 
                for f in os.listdir(self.audio_path) 
                if f.endswith(('.wav', '.mp3', '.flac'))
            ]
        else:
            audio_files = [self.audio_path]
        
        for audio_file in audio_files:
            noise_files = self.extract_noise_from_file(audio_file)
            self.noise_files.extend(noise_files)
        
        self.create_noise_csv()
        return self.noise_files

    def create_noise_csv(self):
        """Create noise.csv file"""
        csv_path = os.path.join(self.output_path, "noise.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, 
                fieldnames=['ID', 'duration', 'wav', 'wav_format', 'wav_opts'])
            writer.writeheader()
            
            for i, noise in enumerate(self.noise_files):
                writer.writerow({
                    'ID': f"Noise_{i+1}",
                    'duration': f"{noise['duration']:.1f}",
                    'wav': noise['path'].replace('\\', '/'),
                    'wav_format': 'wav',
                    'wav_opts': ''
                })
        
        print(f"Created noise.csv at {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract background noise from audio files")
    parser.add_argument("-AUD_PTH", required=True, 
                       help="Path to audio file(s)")
    parser.add_argument("-OUT_PTH", required=True, 
                       help="Output directory")
    parser.add_argument("-PERC", type=float, default=20,
                       help="Noise percentile (1-100)")
    args = parser.parse_args()
    
    extractor = BackgroundNoiseExtractor(
        args.AUD_PTH, 
        args.OUT_PTH,
        noise_percentile=args.PERC
    )
    extractor.process()

if __name__ == "__main__":
    main()