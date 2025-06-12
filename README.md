## TimeDomain  

### SpeedPerturb  

**Effect on Output**  
Resamples the audio signal at a rate similar to the original rate, resulting in a slightly slower or faster signal.  
 
| Parameter     | Description                                                                                      | Default    |
|---------------|--------------------------------------------------------------------------------------------------|------------|
| `orig_freq`   | Sampling rate of the input audio (`16000` or `8000`).                                            | Required   |
| `speeds`      | List of speed percentages: values `<100` speed up, `>100` slow down, `100` keeps original speed. | Required   |

---

### DropChunk  

**Effect on Output**  
Randomly zeros out chunks of the audio signal to encourage the model to rely on all parts of the input instead of specific segments.  
 
| Parameter          | Description                                                                                            | Default |
|--------------------|--------------------------------------------------------------------------------------------------------|---------|
| `drop_length_low`  | Minimum length (in samples) of the chunk to be zeroed out.                                             | `100`   |
| `drop_length_high` | Maximum length (in samples) of the chunk to be zeroed out.                                             | `1000`  |
| `drop_count_low`   | Minimum number of chunks to be zeroed out.                                                             | `1`     |
| `drop_count_high`  | Maximum number of chunks to be zeroed out.                                                             | `3`     |
| `noise_factor`     | Scale factor for adding white noise. `0.0` inserts only zeros, `1.0` keeps average amplitude the same. | `0.0`   |

---

### DropFreq

**Effect on Output**  
Drops a random frequency from the signal to teach models to learn to rely on all parts of the signal, not just a few frequency bands.

| Parameter              | Description                                                                              | Default |
|------------------------|------------------------------------------------------------------------------------------|---------|
| `drop_freq_low`        | The low end of frequencies that can be dropped, as a fraction of the sampling rate / 2.  | `1e-14` |
| `drop_freq_high`       | The high end of frequencies that can be dropped, as a fraction of the sampling rate / 2. | `1`     |
| `drop_freq_count_low`  | The low end of number of frequencies that could be dropped.                              | `1`     |
| `drop_freq_count_high` | The high end of number of frequencies that could be dropped.                             | `3`     |
| `drop_freq_width`      | The width of the frequency band to drop, as a fraction of the sampling_rate / 2.         | `0.05`  |

---

### DoClip

**Effect on Output**  
Mimics audio clipping by clamping the input tensor.

| Parameter   | Description                                              | Default |
|-------------|----------------------------------------------------------|---------|
| `clip_low`  | The low end of amplitudes for which to clip the signal.  | `0.5`   |
| `clip_high` | The high end of amplitudes for which to clip the signal. | `0.5`   |

---

### RandAmp

**Effect on Output**  
Multiples the signal by a random amplitude.

| Parameter  | Description                                  | Default |
|------------|----------------------------------------------|---------|
| `amp_low`  | The minimum amplitude multiplication factor. | `0.5`   |
| `amp_high` | The maximum amplitude multiplication factor. | `1.5`   |

---

### ChannelDrop

**Effect on Output**  
Drops random channels in the multi-channel input waveform.

| Parameter   | Description                | Default |
|-------------|----------------------------|---------|  
| `drop_rate` | The channel dropout factor | `0.1`   |

---

### ChannelSwap

**Effect on Output**  
This function randomly swaps N channels.

| Parameter  | Description                                  | Default |
|------------|----------------------------------------------|---------|
| `min_swap` | The minimum number of channels to swap.      | `0`     |
| `max_swap` | The maximum number of channels to swap.      | `0`     |

---

### CutCat

**Effect on Output**    
Combines segments (with equal length in time) of the time series contained in the batch.

| Parameter          | Description                                 | Default    |
|--------------------|---------------------------------------------|------------|
| `min_num_segments` | The number of segments to combine.          | `Required` |
| `max_num_segments` | The maximum number of segments to combine.  | `Required` |

---

### DropBitResolution

**Effect on Output**    
This class transforms a float32 tensor into a lower resolution one and then converts it back to a float32.
This process loses information and can be used for data augmentation.

| Parameter          | Description                                                                    | Default  |
|--------------------|--------------------------------------------------------------------------------|----------|
| `target_dtype`     | One of "int16", "int8", "float16". If "random", the bit resolution is random.  | `random` |

### SignFlip

**Effect on Output** 
This module negates all the values in a tensor with a given probability. If the sign is not flipped, the original signal is returned unchanged.

| Parameter   | Description                                                   | Default |
|-------------|---------------------------------------------------------------|---------|
| `flip_prob` |   The probability with which to flip the sign of the signal.  | `0.5 `  |

### AddNoise  

**Effect on Output**  
This augmentation adds a noise signal to the input signal to simulate real-world audio conditions.  

**Parameters**  

| Parameter   | Description                                                                                         | Default  |
|-------------|-----------------------------------------------------------------------------------------------------|----------|
| `csv_file`  | Path to a CSV file containing metadata about the noise audio files.                                 | `None`   |
| `pad_noise` | Ensures the noise spans the entire input audio file if set to `True`.                               | `False`  |
| `sorting`   | Order in which to iterate through the CSV file: `random`, `original`, `ascending`, or `descending`. | `random` |
| `snr_low`   | Lower bound for the signal-to-noise ratio (SNR) in decibels.                                        | `0`      |
| `snr_high`  | Upper bound for the signal-to-noise ratio (SNR) in decibels.                                        | `0`      |

---

**CSV Format**  
The CSV file should follow the format below, where each row specifies a noise audio file and its metadata:

| Column      | Description                                    |
|-------------|------------------------------------------------|
| `ID`        | Unique identifier for the noise file.          |
| `duration`  | Duration of the noise file in seconds.         |
| `wav`       | Path to the noise file.                        |
| `wav_format`| Format of the audio file (e.g., `wav`).        |
| `wav_opts`  | Additional options for loading the audio file. |

**Example CSV**  
```csv
ID,duration,wav,wav_format,wav_opts
Noise_1,5.4,./Noise/Noise_1.wav,wav,
Noise_2,3.2,./Noise/Noise_2.wav,wav,
```

### AddRev  

**Effect on Output**  
This augmentation applies reverberation to the input signal using a Room Impulse Response (RIR) file.  

**Parameters**  

| Parameter   | Description                                         | Default  |
|-------------|-----------------------------------------------------|----------|
| `csv_file`  | Path to a CSV file containing metadata about the RIR audio files. | `None`   |

---

**CSV Format**  
The CSV file should follow the format below, where each row specifies an RIR audio file and its metadata:

| Column      | Description                                         |
|-------------|-----------------------------------------------------|
| `ID`        | Unique identifier for the RIR file.                |
| `duration`  | Duration of the RIR file in seconds.               |
| `wav`       | Path to the RIR file.                              |
| `wav_format`| Format of the audio file (e.g., `wav`).            |
| `wav_opts`  | Additional options for loading the audio file.     |

**Example CSV**  
```csv
ID,duration,wav,wav_format,wav_opts
rir1,1.0,./Reverb/rir1.wav,wav,
rir2,0.8,./Reverb/rir2.wav,wav,
```

## Augmenter  

**Effect on Output**  
The `Augmenter` class applies a pipeline of data augmentation techniques, either sequentially or in parallel, to enhance the input dataset.

**Parameters**  

| Parameter               | Description                                                                                                 | Default |
|-------------------------|-------------------------------------------------------------------------------------------------------------|---------|
| `parallel_augment`      | `False`: augmentations are sequential. `True`: augmentations are parallel and concatenated.                 | `False` |
| `concat_original`       | `False`: Output is all augmented data. `True`: Output is a random 50/50 mix of original and augmented data. | `False` |
| `min_augmentations`     | Minimum number of augmentations to apply to the input.                                                      | `1`     |
| `max_augmentations`     | Maximum number of augmentations to apply to the input.                                                      | `1`     |
| `shuffle_augmentations` | If `True`, shuffles the order of augmentations applied to the input.                                        | `False` |
| `repeat_augment`        | Number of times the augmentation pipeline should be repeated on the input.                                  | `1`     |

## Creating the Configuration File  

To use the augmentation pipeline, a configuration file in YAML format must be created. The file is structured into high-level classes that group related augmentations, such as `TimeDomain` for time-based augmentations and `Augmenter` for the augmentation control settings. Below is an example configuration and explanation of its structure.

### Example Configuration  

```yaml
TimeDomain:  
  AddNoise:  
    csv_file: ./noise.csv # Can be a relative or full file path.
    pad_noise: True  
    snr_low: 5  
    snr_high: 20  

  DropChunk:  
    drop_length_low: 200  
    drop_length_high: 1000  
    drop_count_low: 1  
    drop_count_high: 3  
    noise_factor: 0.1  

  SpeedPerturb:  
    orig_freq: 16000  
    speeds: [90, 100, 110]  

Augmenter:  
  parallel_augment: False  
  concat_original: False  
  min_augmentations: 1  
  max_augmentations: 2  
  shuffle_augmentations: True  
  repeat_augment: 1  
```

## Running the script

```powershell
python .\Scripts\aug.py  -AUD_PTH .\Audio\ -CONF_PTH .\config.yml -OUT_PTH .\Output\
```