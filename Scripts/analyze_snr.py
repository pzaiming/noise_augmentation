import os
import numpy as np
import librosa
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_snr(audio_path):
    """Calculate Signal-to-Noise Ratio (SNR) for an audio file"""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate signal power
        signal_power = np.mean(y**2)
        
        # Estimate noise power using frame analysis
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop
        
        # Calculate frame energies
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.mean(frames**2, axis=0)
        
        # Use the lowest 10% of frame energies as noise estimate
        noise_power = np.mean(np.sort(frame_energies)[:len(frame_energies)//10])
        
        # Calculate SNR
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100  # arbitrary high value for cases with very little noise
            
        return max(min(snr, 50), -50)  # Clamp between -50 and 50 dB
        
    except Exception as e:
        print(f"Error calculating SNR for {audio_path}: {str(e)}")
        return None

def analyze_directory(input_dir, output_file):
    """Analyze all audio files in a directory and its subdirectories"""
    # Find all audio files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    
    results = []
    print(f"\nAnalyzing {len(audio_files)} audio files...")
    
    # Process each file
    for audio_path in tqdm(audio_files):
        snr = calculate_snr(audio_path)
        if snr is not None:
            results.append({
                'file': os.path.relpath(audio_path, input_dir),
                'snr': snr
            })
    
    # Create DataFrame and sort by SNR
    df = pd.DataFrame(results)
    df = df.sort_values('snr')
    
    # Save results to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Generate statistics
    print("\nSNR Statistics:")
    print(f"Min SNR: {df['snr'].min():.2f} dB")
    print(f"Max SNR: {df['snr'].max():.2f} dB")
    print(f"Mean SNR: {df['snr'].mean():.2f} dB")
    print(f"Median SNR: {df['snr'].median():.2f} dB")
    
    # Create SNR distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(df['snr'], bins=50, edgecolor='black')
    plt.title('SNR Distribution')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Number of Files')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.splitext(output_file)[0] + '_distribution.png'
    plt.savefig(plot_path)
    print(f"\nSNR distribution plot saved to {plot_path}")
    
    # Group files by SNR ranges
    ranges = np.arange(np.floor(df['snr'].min()), np.ceil(df['snr'].max()) + 10, 10)
    df['snr_range'] = pd.cut(df['snr'], bins=ranges, labels=[f"{ranges[i]:.0f} to {ranges[i+1]:.0f}" for i in range(len(ranges)-1)])
    
    # Print files in each range
    range_output = os.path.splitext(output_file)[0] + '_ranges.txt'
    with open(range_output, 'w') as f:
        for snr_range in df['snr_range'].unique():
            files_in_range = df[df['snr_range'] == snr_range]
            f.write(f"\nSNR Range: {snr_range} dB ({len(files_in_range)} files)\n")
            f.write("-" * 50 + "\n")
            for _, row in files_in_range.iterrows():
                f.write(f"{row['file']}: {row['snr']:.2f} dB\n")
    
    print(f"\nDetailed SNR ranges saved to {range_output}")

def main():
    parser = argparse.ArgumentParser(description='Analyze SNR values of audio files')
    parser.add_argument('--input', required=True, help='Input directory containing audio files')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Convert paths to absolute
    input_dir = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Analyze files
    analyze_directory(input_dir, output_file)

if __name__ == "__main__":
    main()
