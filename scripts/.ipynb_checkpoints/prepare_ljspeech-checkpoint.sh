#!/bin/bash

data_root=$1
tgt_dir=$2
model=$3

function error
{
    if [ -z "$1" ]
    then
        message="fatal error"
    else
        message="fatal error: $1"
    fi

    echo $message
    echo "finished at $(date)"
    exit 1
}

set -eu

stage=3
stop_stage=100
# Convert the wav files to 16 kHz, 16-bit and remove the silences
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    data_root_16khz=$data_root/wavs_16khz
    if [ ! -d ${data_root_16khz} ]; then
        mkdir -p ${data_root_16khz}
        for fpath in $(find ${data_root}/wavs -name "*.wav"); do
            fn=${fpath##*/}
            sox ${fpath} -r 16000 -c 1 -b 16 ${data_root_16khz}/${fn}
        done
    fi
    python scripts/vads.py \
        -r $RVAD_ROOT < ${tgt_dir}/with_silence/train.tsv > ${tgt_dir}/with_silence/train.vads \
        || error "vad failed"
    python scripts/remove_silence.py \
        --tsv ${tgt_dir}/with_silence/train.tsv \
        --vads with_silence/train.vads \
        --out ${tgt_dir}/without_silence/wavs \
        || error "remove silence failed"
fi

# Extract LJSpeech audio features
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    cp $tgt_dir/with_silence/*.tsv $tgt_dir/without_silence
    zsh scripts/prepare_audio.sh \
        $tgt_dir/without_silence \
        $tgt_dir/without_silence/feat \
        $model 512 14
fi

# Extract LJSpeech text transcripts
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then 
    for x in train valid test; do
        python scripts/convert_metadata_ljspeech.py $data_root/metadata.csv $tgt_dir/without_silence/${x}.tsv $tgt_dir/without_silence/${x}.wrd
        python scripts/separate_chars.py $tgt_dir/without_silence/${x}.wrd $tgt_dir/without_silence/${x}.phn
    done
fi

# Process generated text
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then 
    zsh scripts/prepare_text_ljspeech.sh $tgt_dir/without_silence
fi
