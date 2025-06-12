python3 ./Scripts/train.py \
    -GND_PTH ./Ground/ \
    -AUD_PTH ./Audio/ \
    -BASE_MOD_PTH ./Models_Configs/pyannote_base.bin \
    -TRAIN_CONF_PTH ./Models_Configs/training_config.yml \
    -OUT_PTH ./Models_Configs/
mkdir ./No_Augments/
python3 ./Scripts/inference.py \
    -AUD_PTH ./Audio/TEST \
    -MOD_PTH ./Models_Configs/pyannote_trained.ckpt \
    -CONF_PTH ./Models_Configs/pyannote_trained_config.yml \
    -OUT_PTH ./No_Augments/
python3 ./Scripts/eval.py \
    -GND_PTH ./Ground/ \
    -HYP_PTH ./No_Augments/ \
    -OUT_PTH ./
mv ./eval_results.txt ./no_aug_results.txt
rm -rf ./work