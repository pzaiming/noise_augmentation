import os
import argparse
import numpy as np
import librosa
import soundfile as sf
import csv
from utils import path_builder as pb, file_check, rm_ext


class Noise_Extractor:
    def __init__(self, AUD_PTH, OUT_PTH, csv_path=None):
        self.CWD = os.getcwd()  # Instantiate current working directory.
        self.AUD_PTH = pb(self.CWD, file_check(AUD_PTH))
        self.OUT_PTH = pb(self.CWD, OUT_PTH)  # Ensure output directory exists.
        self.csv_path = csv_path if csv_path else os.path.join(self.CWD, "noise.csv")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUT_PTH, exist_ok=True)
        
        # Create Noise subdirectory if it doesn't exist
        self.noise_dir = os.path.join(self.OUT_PTH, "Noise")
        os.makedirs(self.noise_dir, exist_ok=True)

    def plot_frequency_distribution(self):
        """
        Generate frequency distribution plots (ignoring time) for all audio files in the directory.
        Save the plots as images in the output directory.
        """
        audio_files = [f for f in os.listdir(self.AUD_PTH) if f.endswith(('.wav', '.mp3'))]
        for audio_file in audio_files:
            file_path = os.path.join(self.AUD_PTH, audio_file)
            output_image_path = os.path.join(self.OUT_PTH, f"{os.path.splitext(audio_file)[0]}_freq_dist.png")

            try:
                # Load audio file
                y, sr = librosa.load(file_path)

                # Compute the FFT (Fast Fourier Transform)
                fft = np.fft.fft(y)
                frequencies = np.fft.fftfreq(len(fft), 1 / sr)
                magnitude = np.abs(fft)

                # Take only the positive frequencies
                positive_freqs = frequencies[frequencies > 0]
                positive_magnitude = magnitude[frequencies > 0]

                # Plot frequency distribution
                plt.figure(figsize=(10, 6))
                plt.hist(positive_freqs, bins=500, weights=positive_magnitude, color='blue', alpha=0.7)
                plt.title(f"Frequency Distribution for {audio_file}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude")
                plt.grid(True)
                plt.tight_layout()

                # Save the plot
                plt.savefig(output_image_path)
                plt.close()
                print(f"Saved frequency distribution plot for {audio_file} to {output_image_path}")
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    def extract_noise(self, frame_length=2048, hop_length=512, top_db=40, min_silence_duration=0.5):
        """
        Extract background noise from audio files by identifying silent/low-energy segments.
        
        Args:
            frame_length: Frame size for the STFT calculation
            hop_length: Hop size for the STFT calculation
            top_db: The threshold (in decibels) below reference to consider as silence
            min_silence_duration: Minimum duration (in seconds) of silence to extract
            
        Returns:
            List of paths to the extracted noise files
        """
        audio_files = [f for f in os.listdir(self.AUD_PTH) if f.endswith(('.wav', '.mp3'))]
        noise_files = []
        
        for i, audio_file in enumerate(audio_files):
            file_path = os.path.join(self.AUD_PTH, audio_file)
            noise_file_name = f"Noise_{i+1}.wav"
            noise_file_path = os.path.join(self.noise_dir, noise_file_name)
            
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=None)
                
                # Find silent intervals
                intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
                
                # Convert intervals to seconds
                intervals_seconds = intervals / sr
                
                # Find gaps (silence) between intervals
                noise_segments = []
                for i in range(len(intervals) - 1):
                    gap_start = intervals[i][1]
                    gap_end = intervals[i+1][0]
                    gap_duration = (gap_end - gap_start) / sr
                    
                    if gap_duration >= min_silence_duration:
                        noise_segments.append(y[gap_start:gap_end])
                
                # Also check the beginning and end of the file
                if intervals[0][0] > 0:
                    start_silence = y[0:intervals[0][0]]
                    if len(start_silence) / sr >= min_silence_duration:
                        noise_segments.insert(0, start_silence)
                
                if intervals[-1][1] < len(y):
                    end_silence = y[intervals[-1][1]:]
                    if len(end_silence) / sr >= min_silence_duration:
                        noise_segments.append(end_silence)
                
                # If no suitable noise segments found, try to extract low-energy frames
                if not noise_segments:
                    # Compute STFT
                    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
                    
                    # Compute the energy of each frame
                    energy = np.sum(S**2, axis=0)
                    
                    # Find frames with energy below 25th percentile
                    low_energy_threshold = np.percentile(energy, 25)
                    low_energy_frames = np.where(energy < low_energy_threshold)[0]
                    
                    # Group consecutive frames
                    groups = []
                    for k, g in groupby(enumerate(low_energy_frames), lambda x: x[0] - x[1]):
                        groups.append(list(map(lambda x: x[1], g)))
                    
                    # Extract audio from the longest group
                    if groups:
                        longest_group = max(groups, key=len)
                        start_sample = longest_group[0] * hop_length
                        end_sample = (longest_group[-1] + 1) * hop_length
                        noise_segments = [y[start_sample:end_sample]]
                
                # If we found noise segments, concatenate and save
                if noise_segments:
                    noise = np.concatenate(noise_segments)
                    
                    # Save noise file
                    sf.write(noise_file_path, noise, sr)
                    noise_files.append(noise_file_path)
                    
                    print(f"Extracted noise from {audio_file} and saved to {noise_file_path}")
                else:
                    print(f"No suitable silence found in {audio_file}")
            
            except Exception as e:
                print(f"Error extracting noise from {audio_file}: {e}")
        
        return noise_files

    def update_noise_csv(self, noise_files):
        """
        Update or create the noise.csv file with information about the extracted noise files.
        
        Args:
            noise_files: List of paths to the extracted noise files
        """
        # Check if CSV file exists
        csv_exists = os.path.exists(self.csv_path)
        
        # Open CSV file in append mode
        with open(self.csv_path, 'w' if not csv_exists else 'a', newline='') as csvfile:
            fieldnames = ['ID', 'duration', 'wav', 'wav_format', 'wav_opts']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not csv_exists:
                writer.writeheader()
            
            # Write noise file information
            for noise_file in noise_files:
                noise_name = os.path.basename(noise_file)
                noise_id = os.path.splitext(noise_name)[0]
                
                # Get duration using librosa
                y, sr = librosa.load(noise_file, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Use relative path for the CSV
                rel_path = os.path.join("./Noise", noise_name)
                
                writer.writerow({
                    'ID': noise_id,
                    'duration': f"{duration:.1f}",
                    'wav': rel_path,
                    'wav_format': 'wav',
                    'wav_opts': ''
                })
        
        print(f"Updated noise.csv file at {self.csv_path}")

    def process(self):
        """
        Extract noise from audio files and update the noise.csv file.
        """
        noise_files = self.extract_noise()
        if noise_files:
            self.update_noise_csv(noise_files)
            return True
        return False


def extract_noise_from_segments(audio_path, out_dir, segment_length=15, overlap=5):
    """Extract noise from different segments of a long audio file"""
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Get file name without extension
    filename = os.path.basename(audio_path)
    basename = rm_ext(filename)
    
    # Calculate number of segments
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Only segment if the file is longer than twice the segment length
    if duration > segment_length * 2:
        # Calculate segment size in samples
        segment_samples = segment_length * sr
        overlap_samples = overlap * sr
        hop_length = segment_samples - overlap_samples
        
        # Extract noise from each segment
        noise_files = []
        segment_idx = 0
        
        for start in range(0, len(y) - segment_samples, hop_length):
            # Extract segment
            segment = y[start:start + segment_samples]
            
            # Find quietest part of this segment
            segment_noise = extract_noise_from_audio(segment, sr)
            
            # Save the noise segment
            noise_filename = f"{basename}_seg{segment_idx}.wav"
            noise_path = os.path.join(out_dir, noise_filename)
            sf.write(noise_path, segment_noise, sr)
            
            # Record the file
            noise_files.append({
                'path': noise_path,
                'duration': librosa.get_duration(y=segment_noise, sr=sr),
                'start_time': start / sr,
                'segment_idx': segment_idx
            })
            
            segment_idx += 1
            
        return noise_files
    else:
        # For shorter files, just extract one noise sample
        noise_data = extract_noise_from_audio(y, sr)
        noise_path = os.path.join(out_dir, f"{basename}_noise.wav")
        sf.write(noise_path, noise_data, sr)
        
        return [{
            'path': noise_path,
            'duration': librosa.get_duration(y=noise_data, sr=sr),
            'start_time': 0,
            'segment_idx': 0
        }]

def extract_noise_from_audio(audio_data, sr, frame_len=0.5, percentile=10):
    """Extract the quietest parts of the audio as noise"""
    
    # Calculate frame size in samples
    frame_size = int(frame_len * sr)
    
    # Calculate energy for each frame
    energies = []
    for i in range(0, len(audio_data) - frame_size, frame_size // 2):
        frame = audio_data[i:i + frame_size]
        energy = np.sum(frame**2) / len(frame)
        energies.append((energy, i))
    
    # Sort frames by energy
    energies.sort(key=lambda x: x[0])
    
    # Take the frames with lowest energy (based on percentile)
    quiet_frames_count = max(1, int(len(energies) * percentile / 100))
    quiet_frames = energies[:quiet_frames_count]
    
    # Concatenate these frames to create noise sample
    noise_data = np.zeros(frame_size * quiet_frames_count)
    for i, (_, start_idx) in enumerate(quiet_frames):
        frame = audio_data[start_idx:start_idx + frame_size]
        if len(frame) == frame_size:  # Avoid incomplete frames
            noise_data[i * frame_size:(i + 1) * frame_size] = frame
    
    return noise_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-AUD_PTH", help="Path to a directory containing audio files to extract noise from.", required=True)
    parser.add_argument("-OUT_PTH", help="Path to an output directory.", required=True)
    parser.add_argument("-CSV_PTH", help="Path to noise.csv file (optional).")
    parser.add_argument("--plot", action="store_true", help="Generate frequency distribution plots.")
    args = parser.parse_args()

    # Instantiate Noise_Extractor and process files
    extractor = Noise_Extractor(args.AUD_PTH, args.OUT_PTH, args.CSV_PTH)
    
    if args.plot:
        extractor.plot_frequency_distribution()
    
    extractor.process()


# Add missing import for groupby
from itertools import groupby

if __name__ == "__main__":
    main()