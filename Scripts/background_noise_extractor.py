import os
import numpy as np
import librosa
import soundfile as sf
import csv
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.preprocessing import StandardScaler

class BackgroundNoiseExtractor:
    def __init__(self, audio_path, output_path, 
                 frame_size=0.1, 
                 noise_threshold=0.02,
                 min_segment_duration=1.0,
                 noise_percentile=20):
        self.audio_path = audio_path
        self.output_path = output_path
        self.noise_dir = os.path.join(output_path, "Noise")
        os.makedirs(self.noise_dir, exist_ok=True)
        
        self.frame_size = frame_size
        self.noise_threshold = noise_threshold
        self.min_segment_duration = min_segment_duration
        self.noise_percentile = noise_percentile
        self.similarity_threshold = 0.85
        self.noise_files = []  # Initialize the list to store noise files

    def detect_noise_segments(self, audio_data, sr):
        """Detect distinct noise segments using spectral clustering"""
        # Calculate spectrogram
        n_fft = int(self.frame_size * sr)
        hop_length = n_fft // 4
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)

        # Calculate frame-wise features
        features = []
        frame_energies = np.mean(magnitude, axis=0)
        
        for i in range(magnitude.shape[1]):
            # Safe calculation of spectral centroid
            magnitude_sum = np.sum(magnitude[:, i])
            if magnitude_sum > 1e-10:  # Avoid division by zero
                spec_centroid = np.sum(magnitude[:, i] * np.arange(len(magnitude[:, i]))) / magnitude_sum
            else:
                spec_centroid = 0
                
            # Safe calculation of spectral flatness
            mean_magnitude = np.mean(magnitude[:, i])
            if mean_magnitude > 1e-10:  # Avoid division by zero
                spec_flatness = np.exp(np.mean(np.log(magnitude[:, i] + 1e-10))) / mean_magnitude
            else:
                spec_flatness = 0
                
            # Only add valid features
            if np.isfinite([frame_energies[i], spec_centroid, spec_flatness]).all():
                features.append([frame_energies[i], spec_centroid, spec_flatness])

        features = np.array(features)
        
        # Check if we have enough valid features
        if len(features) < 2:
            print(f"Not enough valid features detected, using single noise profile")
            return None, 1

        # Normalize features with robust scaling
        features_scaled = np.zeros_like(features)
        for i in range(features.shape[1]):
            column = features[:, i]
            median = np.median(column)
            iqr = np.percentile(column, 75) - np.percentile(column, 25)
            if iqr > 0:
                features_scaled[:, i] = (column - median) / iqr
            else:
                features_scaled[:, i] = column - median

        # Perform clustering
        Z = linkage(features_scaled, method='ward')
        clusters = fcluster(Z, t=self.similarity_threshold, criterion='distance')
        unique_clusters = np.unique(clusters)
        
        # Filter out too short segments
        valid_clusters = []
        for cluster in unique_clusters:
            cluster_frames = np.where(clusters == cluster)[0]
            cluster_duration = len(cluster_frames) * hop_length / sr
            if cluster_duration >= self.min_segment_duration:
                valid_clusters.append(cluster)
        
        return clusters, len(valid_clusters)

    def validate_noise_profile(self, profile, threshold=0.01):
        """Validate that noise profile has sufficient energy"""
        return np.mean(np.abs(profile)) > threshold

    def extract_noise_profile(self, audio_data, sr, segment_indices=None):
        """Extract noise profile from specified segment with validation"""
        # If no segment specified, use entire audio
        if segment_indices is None:
            segment_indices = np.arange(len(audio_data))
            
        segment = audio_data[segment_indices]
        
        # Skip if segment is too quiet
        if np.mean(np.abs(segment)) < 0.01:
            return None, None
        
        # Calculate noise profile
        n_fft = int(self.frame_size * sr)
        D = librosa.stft(segment, n_fft=n_fft)
        magnitude = np.abs(D)
        
        # Use median to get stable noise profile
        noise_profile = np.median(magnitude, axis=1)
        
        # Validate the profile
        if not self.validate_noise_profile(noise_profile):
            return None, None
        
        return noise_profile, n_fft

    def synthesize_noise(self, profile, n_fft, sr, duration=2.0):
        """Synthesize noise from spectral profile"""
        # Calculate number of frames needed
        hop_length = n_fft // 4
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
        """Extract one or more noise profiles based on content with silence checking"""
        print(f"\nProcessing {audio_file}...")
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        
        # Detect noise segments
        clusters, n_profiles = self.detect_noise_segments(y, sr)
        print(f"Detected {n_profiles} distinct noise profile(s)")
        
        extracted_noise = []
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        
        if clusters is not None and n_profiles > 1:
            # Extract each distinct noise profile
            valid_profiles = 0
            for i in range(n_profiles):
                segment_indices = np.where(clusters == i + 1)[0]
                profile, n_fft = self.extract_noise_profile(y, sr, segment_indices)
                
                # Skip if profile is invalid
                if profile is None:
                    continue
                    
                # Generate noise from profile
                noise = self.synthesize_noise(profile, n_fft, sr)
                
                # Validate generated noise
                if np.mean(np.abs(noise)) < 0.01:
                    continue
                
                # Save valid noise file
                valid_profiles += 1
                noise_filename = f"{basename}_noise_{valid_profiles}.wav"
                noise_path = os.path.join(self.noise_dir, noise_filename)
                sf.write(noise_path, noise, sr)
                
                extracted_noise.append({
                    'path': noise_path,
                    'filename': noise_filename,
                    'duration': len(noise) / sr,
                    'original_file': os.path.basename(audio_file),
                    'profile_index': valid_profiles
                })
                print(f"Created valid noise profile {valid_profiles} from {basename}")
        else:
            # Extract single noise profile
            profile, n_fft = self.extract_noise_profile(y, sr)
            
            # Only create noise file if profile is valid
            if profile is not None:
                noise = self.synthesize_noise(profile, n_fft, sr)
                
                # Check if generated noise is valid
                if np.mean(np.abs(noise)) > 0.01:
                    noise_filename = f"{basename}_noise.wav"
                    noise_path = os.path.join(self.noise_dir, noise_filename)
                    sf.write(noise_path, noise, sr)
                    
                    extracted_noise.append({
                        'path': noise_path,
                        'filename': noise_filename,
                        'duration': len(noise) / sr,
                        'original_file': os.path.basename(audio_file)
                    })
                    print(f"Created valid single noise profile from {basename}")
                else:
                    print(f"Skipping {basename} - generated noise too quiet")
            else:
                print(f"Skipping {basename} - invalid noise profile")
        
        return extracted_noise

    def process(self):
        """Process all audio files and extract noise"""
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
        
        # Create noise.csv after processing all files
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract background noise from audio files")
    parser.add_argument("-AUD_PTH", required=True, help="Path to audio file(s)")
    parser.add_argument("-OUT_PTH", required=True, help="Output directory")
    parser.add_argument("-THRESHOLD", type=float, default=0.02, 
                        help="Noise detection threshold")
    parser.add_argument("-PERC", type=float, default=20,
                        help="Noise percentile (1-100)")
    args = parser.parse_args()
    
    extractor = BackgroundNoiseExtractor(
        args.AUD_PTH, 
        args.OUT_PTH,
        noise_threshold=args.THRESHOLD,
        noise_percentile=args.PERC
    )
    extractor.process()

if __name__ == "__main__":
    main()