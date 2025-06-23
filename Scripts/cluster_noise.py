import os
import csv
import shutil
import numpy as np
from pathlib import Path
import argparse
import librosa

def read_noise_csv(csv_path):
    """Read noise files and their SNR values from CSV"""
    noise_files = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            wav_path = row['wav']
            
            # Calculate actual SNR from the audio file
            print(f"\rProcessing file {i+1}: {os.path.basename(wav_path)}", end="")
            snr = calculate_snr(wav_path)
            
            noise_files.append({
                'path': wav_path,
                'snr': snr,
                'id': row['ID']
            })
    print("\nFinished processing all files")
    return noise_files

def calculate_snr(audio_path):
    """Calculate SNR for an audio file"""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate signal power
        signal_power = np.mean(y**2)
        
        # Estimate noise power (using a simple method)
        # Here we're using the lowest 10% of frame energies as noise estimate
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop
        
        # Calculate frame energies
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.mean(frames**2, axis=0)
        
        # Use the lowest 10% as noise estimate
        noise_power = np.mean(np.sort(frame_energies)[:len(frame_energies)//10])
        
        # Calculate SNR
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100  # arbitrary high value for cases with very little noise
            
        return max(min(snr, 50), -50)  # Clamp between -50 and 50 dB
        
    except Exception as e:
        print(f"Error calculating SNR for {audio_path}: {str(e)}")
        return 0  # Return 0 as default SNR in case of error

def create_dynamic_clusters(noise_files, snr_range=10.0):
    """Cluster noise files based on SNR values with dynamic range-based clustering"""
    # Sort noise files by SNR
    sorted_files = sorted(noise_files, key=lambda x: x['snr'])
    
    # Find min and max SNR values
    min_snr = min(f['snr'] for f in sorted_files)
    max_snr = max(f['snr'] for f in sorted_files)
    
    # Calculate number of clusters needed
    num_clusters = int(np.ceil((max_snr - min_snr) / snr_range))
    
    # Create cluster boundaries
    cluster_boundaries = [min_snr + i * snr_range for i in range(num_clusters + 1)]
    
    # Initialize clusters
    clusters = []
    for i in range(num_clusters):
        lower_bound = cluster_boundaries[i]
        upper_bound = cluster_boundaries[i + 1]
        
        # Get files within this SNR range
        cluster_files = [
            f for f in sorted_files 
            if lower_bound <= f['snr'] < upper_bound or 
               (i == num_clusters - 1 and f['snr'] >= lower_bound)  # Include upper bound for last cluster
        ]
        
        # Only create cluster if it has files
        if cluster_files:
            avg_snr = np.mean([f['snr'] for f in cluster_files])
            clusters.append({
                'files': cluster_files,
                'avg_snr': avg_snr,
                'snr_range': (lower_bound, upper_bound)
            })
    
    return clusters

def organize_noise_files(clusters, output_base_dir):
    """Organize noise files into cluster directories"""
    # Create base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each cluster
    for i, cluster in enumerate(clusters):
        # Create cluster directory with SNR range information
        lower_bound, upper_bound = cluster['snr_range']
        cluster_dir = os.path.join(
            output_base_dir, 
            f"cluster{i+1}_snr{lower_bound:.1f}to{upper_bound:.1f}"
        )
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Copy files to cluster directory
        for file_info in cluster['files']:
            src_path = file_info['path']
            if os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(cluster_dir, filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {filename} to {cluster_dir}")
            else:
                print(f"Warning: Source file not found: {src_path}")

def main():
    parser = argparse.ArgumentParser(description='Cluster noise files based on SNR values')
    parser.add_argument('--csv', required=True, help='Path to noise.csv file')
    parser.add_argument('--output', required=True, help='Base directory for clustered noise files')
    parser.add_argument('--snr-range', type=float, default=10.0, 
                       help='SNR range for each cluster in dB (default: 10.0)')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    csv_path = os.path.abspath(args.csv)
    output_dir = os.path.abspath(args.output)
    
    print(f"\nReading noise files from: {csv_path}")
    noise_files = read_noise_csv(csv_path)
    print(f"Found {len(noise_files)} noise files")
    
    print(f"\nCreating clusters with {args.snr_range} dB range...")
    clusters = create_dynamic_clusters(noise_files, args.snr_range)
    
    print("\nCluster Statistics:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}:")
        print(f"  Average SNR: {cluster['avg_snr']:.2f} dB")
        print(f"  Number of files: {len(cluster['files'])}")
        print(f"  SNR Range: {min([f['snr'] for f in cluster['files']]):.2f} to {max([f['snr'] for f in cluster['files']]):.2f} dB")
    
    print(f"\nOrganizing files into: {output_dir}")
    organize_noise_files(clusters, output_dir)
    
    print("\nClustering complete!")

if __name__ == "__main__":
    main()
