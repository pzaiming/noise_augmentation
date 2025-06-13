@echo off
setlocal enabledelayedexpansion

echo Starting Audio Augmentation Pipeline...
echo.

:: Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Please ensure Python dependencies are installed.
)

:: Create required directories
if not exist "Extracted_Noise" mkdir "Extracted_Noise"
if not exist "AUG_AUDIO" mkdir "AUG_AUDIO"

:: Clean previous outputs
echo Cleaning previous outputs...
if exist "Extracted_Noise\*" del /Q "Extracted_Noise\*"
if exist "AUG_AUDIO\*" del /Q "AUG_AUDIO\*"

:: Step 1: Extract background noise
echo.
echo Step 1: Extracting background noise...
python Scripts/background_noise_extractor.py -AUD_PTH ./Noise_Source/ -OUT_PTH ./Extracted_Noise
if errorlevel 1 (
    echo Error during noise extraction!
    exit /b 1
)

:: Step 2: Run augmentation
echo.
echo Step 2: Applying noise augmentation...
python Scripts/aug.py -AUD_PTH ./Audio/TRAIN/ -CONF_PTH ./Config/noise_aug_config.yml -OUT_PTH ./AUG_AUDIO/
if errorlevel 1 (
    echo Error during augmentation!
    exit /b 1
)

echo.
echo Augmentation pipeline completed successfully!
echo Augmented files are in: .\AUG_AUDIO\

:: Deactivate virtual environment
if exist .venv\Scripts\activate.bat call deactivate

pause