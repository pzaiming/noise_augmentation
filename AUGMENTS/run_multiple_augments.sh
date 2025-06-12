mkdir -p ./AUG_AUDIO/ 
rm -rf ./AUG_AUDIO/*
python3 ./Scripts/aug.py -AUD_PTH ./Audio/TRAIN/ -CONF_PTH ./aug_conf.yml -OUT_PTH ./AUG_AUDIO/ 
# Augmentations is done proceed. 
# Assuming AUG_AUDIO has directories of augmented audio files, we should now prepare a folder to stage the audio in temporarily. 
# The following code should only be run if running for the first time else comment out. 
mkdir -p ./Staging/
mkdir -p ./Staging/DEV/
mkdir -p ./Staging/TRAIN/
mkdir -p ./Staging/TEST/
mkdir -p ./Staging/Ground/
#cp -r ./Audio/DEV/* ./Staging/DEV/ 
#cp -r ./Audio/TEST/* ./Staging/TEST/ 
mkdir -p ./HYP/ 

for AUG_PATH in ./AUG_AUDIO/*; do
    AUG_NAME=$(basename "$AUG_PATH")
    # Clean directory first in case we have funky files in there leftover.
    rm -rf ./work/*
    rm -rf ./Staging/TRAIN/* 
    rm -rf ./HYP/* 
    rm -rf ./Staging/Ground/*
    # We need to setup the ground truth for the Train and AUG_Train.
    python3 ./Scripts/rewrite_timings.py -AUG_PTH ./AUG_AUDIO/${AUG_NAME}/ -AUD_PTH ./Audio/TRAIN/ -GND_PTH ./Ground/ -OUT_PTH ./Staging/Ground/
    cp -r ./Ground/* ./Staging/Ground/
    # Bring in clean untainted TRAIN data 
    cp -r ./Audio/TRAIN/* ./Staging/TRAIN/ 
    # Bring in the augmented data of choice 
    cp -r ./AUG_AUDIO/${AUG_NAME}/* ./Staging/TRAIN/
    # Staging directory is ready for use. 
    python3 ./Scripts/train.py -AUD_PTH ./Staging/ -GND_PTH ./Staging/Ground/ -BASE_MOD_PTH ./Models_Configs/pyannote_base.bin -TRAIN_CONF_PTH ./Models_Configs/training_config.yml -OUT_PTH ./Models_Configs/ 
    mv ./Models_Configs/pyannote_trained_config.yml ./Models_Configs/pyannote_trained_${AUG_NAME}_config.yml 
    mv ./Models_Configs/pyannote_trained.ckpt ./Models_Configs/pyannote_trained_${AUG_NAME}.ckpt 
    python3 ./Scripts/inference.py -AUD_PTH ./Staging/TEST/ -MOD_PTH ./Models_Configs/pyannote_trained_${AUG_NAME}.ckpt -CONF_PTH ./Models_Configs/pyannote_trained_${AUG_NAME}_config.yml -OUT_PTH ./HYP/ 
    python3 ./Scripts/eval.py -GND_PTH ./Staging/Ground/ -HYP_PTH ./HYP/ -OUT_PTH ./ 
    mv ./eval_results.txt ./${AUG_NAME}_eval_results.txt 
done