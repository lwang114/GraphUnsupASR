#!/bin/bash

source /home/hertin/.bashrc
data_root=$1
tgt_dir=$2
model=$3
max_image_per_ltr=$4
n_cpc_clus=$5

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

stage=4
stop_stage=100
extract_audio_feat=false
extract_cpc_feat=true
cluster_image=true
apply_image_cluster=true

# CPC configs
n_predicts=3
n_neg=128
feat_name=vgg19

# Dataset configs
splits="train valid test"
s=without_silence_CLUS${n_cpc_clus}
lm_dir=$tgt_dir/$s/phones
fst_dir=$tgt_dir/$s/fst/phn_to_phn
if [ ! -d $lm_dir ]; then
    mkdir -p $lm_dir
fi
if [ ! -d $fst_dir ]; then
    mkdir -p $fst_dir
fi

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
        -r $RVAD_ROOT < $tgt_dir/with_silence/train.tsv > $tgt_dir/with_silence/train.vads \
        || error "vad failed"
    python scripts/remove_silence.py \
        --tsv $tgt_dir/with_silence/train.tsv \
        --vads $tgt_dir/with_silence/train.vads \
        --out $tgt_dir/$s/wavs \
        || error "remove silence failed"
fi

# Extract LJSpeech audio features
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    cp $tgt_dir/with_silence/*.tsv $tgt_dir/$s
    if $extract_audio_feat; then
        zsh scripts/prepare_audio.sh \
            $tgt_dir/$s \
            $tgt_dir/$s/feat \
            $model 512 14
    fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for x in ${splits}; do
        python scripts/convert_metadata_ljspeech.py $data_root/metadata.csv $tgt_dir/$s/${x}.tsv $tgt_dir/$s/${x}_gt.wrd
    done
fi

# Train a CPC using fingerspelling sequence
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    if $extract_cpc_feat; then
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate /home/lwang114/.conda/envs/cpc37
        
        cwd=$(pwd)
        cd $CPC_ROOT/cpc
        bash run.swb --max-image-per-ltr $max_image_per_ltr
        cd ${cwd}

        source /home/hertin/.bashrc
        conda activate /home/hertin/.conda/envs/wav2vec 
    fi
fi

# Extract CPC features
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then 
    if $extract_cpc_feat; then
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate /home/lwang114/.conda/envs/cpc37
    fi
    ckpt_path=${CPC_ROOT}/exp/cpc_npredicts_${n_predicts}_${feat_name}_asl_libri960hr/checkpoint_20.pt
    ch2img_path=/home/hertin/manifest/asl_alphabet/ch2img.json
    if [ $max_image_per_ltr -ge 0 ]; then
        if [ $n_neg -lt 128 ]; then
            ckpt_path=${CPC_ROOT}/exp/cpc_npredicts_${n_predicts}_${feat_name}_asl_libri960hr_${n_neg}negatives_${max_image_per_ltr}images_per_ltr/checkpoint_20.pt
        else
            ckpt_path=${CPC_ROOT}/exp/cpc_npredicts_${n_predicts}_${feat_name}_asl_libri960hr_${max_image_per_ltr}images_per_ltr/checkpoint_20.pt
        fi
        ch2img_subset_path=manifest/asl_alphabet/ch2img_${max_image_per_ltr}images_per_ltr.json
        #if [ ! -f $ch2img_subset_path ]; then
        #    python subset_asl_images.py ${ch2img_path} ${ch2img_subset_path} --max_image_per_ltr ${max_image_per_ltr}
        #fi
        ch2img_path=$ch2img_subset_path
    fi

    echo ${tgt_dir}
    for x in test valid train; do
        echo "Extract CPC features for ${x} set"
        python scripts/separate_chars.py ${tgt_dir}/$s/${x}_gt.wrd ${tgt_dir}/$s/${x}.trn 
        if $extract_cpc_feat; then
            python ${CPC_ROOT}/cpc/build_CPC_features.py \
                ${ckpt_path} \
                ${tgt_dir}/$s \
                ${ch2img_path} \
                manifest/asl_alphabet/${feat_name}/train \
                ${tgt_dir}/fs_feat/cpc_npredicts${n_predicts}_${n_neg}negatives_${max_image_per_ltr}images_per_ltr \
                --split ${x} --cpu
        fi
    done
     
    if $extract_cpc_feat; then
        source /home/hertin/.bashrc
        conda activate /home/hertin/.conda/envs/wav2vec 
    fi
fi

# Cluster CPC features
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then 
    tgt_cpc_dir=${tgt_dir}/fs_feat/cpc_npredicts${n_predicts}_${n_neg}negatives_${max_image_per_ltr}images_per_ltr
    if [ $cluster_image = true ]; then
        python scripts/image_cluster_sklearn.py \
            ${tgt_cpc_dir}/train.npy \
            --save-dir ${tgt_cpc_dir} \
            --n_clusters ${n_cpc_clus} --sample-pct 1.0 \
        || error "image_cluster_sklearn.py fails"
    fi

    for split in ${splits}; do
        if [ $apply_image_cluster = true ]; then
            python scripts/image_apply_cluster_sklearn.py \
                ${tgt_cpc_dir} --split ${split} --path ${tgt_cpc_dir}/CLUS${n_cpc_clus} \
            || error "image_apply_cluster_sklearn.py fails"
        fi
        echo "Compute NMI for ${split} set ..."
        python scripts/nmi.py \
            ${tgt_cpc_dir}/CLUS${n_cpc_clus}/${split}.src \
            ${tgt_dir}/${s}/${split}.trn \
            --fmt sklearn \
            || error "nmi.py failed"
   done
fi

# Extract LJSpeech fingerspelling image sequence
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then 
    tgt_cpc_dir=${tgt_dir}/fs_feat/cpc_npredicts${n_predicts}_${n_neg}negatives_${max_image_per_ltr}images_per_ltr
    for x in ${splits}; do
        cp ${tgt_dir}/${s}/${x}.tsv ${tgt_cpc_dir}/CLUS${n_cpc_clus}
        python scripts/create_fingerspell_sequence.py \
            --src_path ${tgt_cpc_dir}/CLUS${n_cpc_clus}/${x}.src \
            --lengths_path ${tgt_cpc_dir}/${x}.lengths \
            --wrd_path ${tgt_dir}/$s/${x}_gt.wrd \
            --out_prefix ${tgt_dir}/$s/$x || error "create_fingerspell_dataset.py fails"  
    done
fi

# Process generated text
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    zsh scripts/prepare_text_fs_ljspeech.sh $tgt_dir $s
fi
