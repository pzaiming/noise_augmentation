FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
WORKDIR /app

# Installation
RUN apt-get update --fix-missing && \
    apt-get install -y build-essential cmake git wget nano vim sox libsox-dev ffmpeg python3-pip && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && rm -rf /var/cache/apt/archives

# Pytorch
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-deps https://github.com/marianne-m/brouhaha-vad/archive/main.zip

COPY ./requirements.txt /app

RUN pip install -r requirements.txt \ 
    pip cache purge

COPY ./Scripts/* /app/Scripts/
COPY ./VAD_Scripts/* /app/Scripts/
COPY ./noise.csv /app/noise.csv
COPY ./run_augment_time_correction.sh /app/run_augment_time_correction.sh
COPY ./run_augment.sh /app/run_augment.sh
COPY ./run_default.sh /app/run_default.sh
COPY ./rand.sh /app/rand.sh
COPY ./Noise/ /app/Noise/
COPY ./Models_Configs/pyannote_base_config.yml /app/Models_Configs/pyannote_base_config.yml
COPY ./Models_Configs/pyannote_base.bin /app/Models_Configs/pyannote_base.bin
COPY ./Models_Configs/training_config.yml /app/Models_Configs/training_config.yml

EXPOSE 8050