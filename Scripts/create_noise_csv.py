import os
import csv
import soundfile as sf
from pathlib import Path

def create_noise_csv(noise_dir, output_path):
    """Create noise.csv for existing noise files in Noise subdirectory"""
    noise_files = []
    
    # Look in the Noise subdirectory
    noise_path = Path(noise_dir) / 'Noise'
    if not noise_path.exists():
        noise_path = Path(noise_dir)  # fallback to main directory
    
    print(f"Scanning for noise files in: {noise_path}")
    
    # Find all audio files recursively
    for file in noise_path.rglob('*'):
        if file.suffix.lower() in ['.wav', '.mp3', '.flac']:
            try:
                # Get audio duration
                info = sf.info(str(file))
                duration = info.duration
                
                noise_files.append({
                    'ID': f'Noise_{len(noise_files)+1}',
                    'duration': f'{duration:.1f}',
                    'wav': str(file.absolute()).replace('\\', '/'),
                    'wav_format': file.suffix[1:],  # remove dot
                    'wav_opts': ''
                })
                print(f"Found noise file: {file.name}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    if not noise_files:
        print("No noise files found! Please check:")
        print(f"1. Files exist in {noise_path}")
        print("2. Files have correct extensions (.wav, .mp3, .flac)")
        return
    
    # Write CSV file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, 
            fieldnames=['ID', 'duration', 'wav', 'wav_format', 'wav_opts'])
        writer.writeheader()
        writer.writerows(noise_files)
    
    print(f"\nCreated noise.csv with {len(noise_files)} entries at:")
    print(output_path.absolute())

if __name__ == '__main__':
    create_noise_csv('./Extracted_Noise', './Extracted_Noise/noise.csv')