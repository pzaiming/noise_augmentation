import os
import argparse
import subprocess
import shutil
import sys
import uuid
import csv
from utils import path_builder as pb, file_check

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    # Use list form for subprocess to avoid shell escaping issues
    if isinstance(command, str):
        import shlex
        command = shlex.split(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
    
    return process.returncode == 0

def generate_noise_csv(noise_dir, csv_path):
    """Create a CSV file with entries for all noise files in the given directory"""
    if not os.path.exists(noise_dir):
        print(f"Warning: Noise directory not found at {noise_dir}")
        return False
    
    # Get list of noise files in the directory
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    
    if not noise_files:
        print(f"Warning: No noise files found in {noise_dir}")
        return False
    
    print(f"Found {len(noise_files)} noise files in {noise_dir}")
    
    # Create the CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'duration', 'wav', 'wav_format', 'wav_opts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each noise file
        for i, noise_file in enumerate(noise_files):
            # Create a unique ID
            unique_id = f"NoiseAug_{i}_{uuid.uuid4().hex[:8]}"
            
            # Get the full path to the noise file and use forward slashes
            file_path = os.path.join(noise_dir, noise_file)
            file_path = file_path.replace('\\', '/')
            
            # Get duration using librosa or assume a default
            try:
                import librosa
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
            except:
                # If librosa fails or isn't available, use a default duration
                duration = 5.0
                print(f"Warning: Could not determine duration for {noise_file}, using default")
            
            writer.writerow({
                'ID': unique_id,
                'duration': f"{duration:.1f}",
                'wav': file_path,
                'wav_format': 'wav',
                'wav_opts': ''
            })
            print(f"Added {noise_file} to noise.csv (ID: {unique_id})")
    
    print(f"Created noise CSV at {csv_path}")
    return True

def auto_noise_augment(noise_source_path, target_audio_path, output_path, config_template_path=None, use_vad=True):
    """
    Automated pipeline to extract noise from one audio source and use it to augment target audio
    
    Args:
        noise_source_path: Path to audio file(s) to extract noise from
        target_audio_path: Path to audio file(s) to augment with extracted noise
        output_path: Path for output files
        config_template_path: Path to a config template (optional)
        use_vad: Whether to use VAD-based noise extraction (True) or simple extraction (False)
    """
    # Get the Python interpreter path that was used to run this script
    python_path = sys.executable
    
    # Create necessary directories
    cwd = os.getcwd()
    extracted_noise_dir = os.path.join(cwd, "Extracted_Noise")
    config_dir = os.path.join(cwd, "Config")
    
    # Create directories if they don't exist
    os.makedirs(extracted_noise_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # Path to the noise CSV and config file
    noise_csv_path = os.path.join(config_dir, "noise.csv")
    config_path = os.path.join(config_dir, "noise_aug_config.yml")
    
    # Step 1: Extract noise from source audio to Extracted_Noise
    print("\n--- Step 1: Extracting noise from source audio ---")
    
    if use_vad:
        # Use Silero VAD-based noise extraction
        extract_cmd = [python_path, "Scripts/silero_vad_extractor.py", 
                      "-AUD_PTH", noise_source_path, 
                      "-OUT_PTH", extracted_noise_dir]
    else:
        # Use traditional noise extraction
        extract_cmd = [python_path, "Scripts/noise.py", 
                      "-AUD_PTH", noise_source_path, 
                      "-OUT_PTH", extracted_noise_dir]
    
    if not run_command(extract_cmd):
        print("Error extracting noise. Exiting.")
        return False
    
    # Use the noise.csv file created by the extractor
    if os.path.exists(os.path.join(extracted_noise_dir, "noise.csv")):
        shutil.copy(os.path.join(extracted_noise_dir, "noise.csv"), noise_csv_path)
    else:
        # Step 2: Create noise.csv and config file
        print("\n--- Step 2: Creating augmentation configuration ---")
        if not generate_noise_csv(os.path.join(extracted_noise_dir, "Noise"), noise_csv_path):
            print("Error: Could not create noise CSV")
            return False
    
    # Create config file
    if config_template_path and os.path.exists(config_template_path):
        shutil.copy(config_template_path, config_path)
        print(f"Using config template from {config_template_path}")
    else:
        with open(config_path, "w") as f:
            f.write(f"""TimeDomain:
  AddNoise:
    csv_file: {noise_csv_path}
    pad_noise: True
    snr_low: 5
    snr_high: 15
    
Augmenter:
  parallel_augment: True
  concat_original: False
  min_augmentations: 1
  max_augmentations: 1
  shuffle_augmentations: False
  repeat_augment: 1
""")
        print(f"Created default config at {config_path}")
    
    # Step 3: Run augmentation using noise from Extracted_Noise
    print("\n--- Step 3: Applying noise augmentation to target audio ---")
    augment_cmd = [python_path, "Scripts/aug.py", 
                  "-AUD_PTH", target_audio_path, 
                  "-CONF_PTH", config_path, 
                  "-OUT_PTH", output_path]
    if not run_command(augment_cmd):
        print("Error during augmentation. Exiting.")
        return False
    
    print(f"\nAugmentation complete! Augmented files are in: {output_path}")
    
    # Copy noise CSV to output directory for reference
    if os.path.exists(noise_csv_path):
        os.makedirs(os.path.join(output_path, "config"), exist_ok=True)
        shutil.copy(noise_csv_path, os.path.join(output_path, "config", "noise_used.csv"))
        shutil.copy(config_path, os.path.join(output_path, "config", "config_used.yml"))
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract noise from one audio source and use it to augment target audio")
    parser.add_argument("-NOISE_SRC", help="Path to audio file(s) to extract noise from", required=True)
    parser.add_argument("-TARGET", help="Path to audio file(s) to augment with the extracted noise", required=True)
    parser.add_argument("-OUT_PTH", help="Path for output files", required=True)
    parser.add_argument("-CONF_TPL", help="Path to a config template (optional)")
    parser.add_argument("--use-vad", action="store_true", help="Use VAD-based noise extraction")
    args = parser.parse_args()
    
    auto_noise_augment(args.NOISE_SRC, args.TARGET, args.OUT_PTH, args.CONF_TPL, args.use_vad)

if __name__ == "__main__":
    main()