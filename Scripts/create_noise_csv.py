import os
import csv
import librosa
import argparse

def create_noise_csv(noise_dir, csv_path):
    """
    Create a single noise.csv file that references all noise files in the given directory.
    
    Args:
        noise_dir: Directory containing noise audio files
        csv_path: Path to create/update the noise.csv file
    """
    # Find all audio files in the noise directory
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not noise_files:
        print(f"No audio files found in {noise_dir}")
        return False
    
    print(f"Found {len(noise_files)} audio files in {noise_dir}")
    
    # Create/update the CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'duration', 'wav', 'wav_format', 'wav_opts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each noise file
        for i, noise_file in enumerate(noise_files):
            file_path = os.path.join(noise_dir, noise_file)
            noise_id = f"Noise_{i+1}"
            
            try:
                # Get duration using librosa
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Get file format
                wav_format = os.path.splitext(noise_file)[1][1:].lower()
                
                # Use relative path for the CSV
                rel_path = os.path.join("./Extracted_Noise", noise_file)
                
                writer.writerow({
                    'ID': noise_id,
                    'duration': f"{duration:.1f}",
                    'wav': rel_path,
                    'wav_format': wav_format,
                    'wav_opts': ''
                })
                print(f"Added {noise_file} to noise.csv (ID: {noise_id}, duration: {duration:.1f}s)")
            except Exception as e:
                print(f"Error processing {noise_file}: {e}")
    
    print(f"Successfully created {csv_path} with {len(noise_files)} noise samples")
    return True

def main():
    parser = argparse.ArgumentParser(description="Create a single noise.csv file for all noise samples")
    parser.add_argument("-NOISE_DIR", default="./Noise_source", help="Directory containing noise audio files")
    parser.add_argument("-CSV_PATH", default="./noise.csv", help="Path to create the noise.csv file")
    args = parser.parse_args()
    
    create_noise_csv(args.NOISE_DIR, args.CSV_PATH)

if __name__ == "__main__":
    main()