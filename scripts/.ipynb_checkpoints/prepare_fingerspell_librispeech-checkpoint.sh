#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
tgt_dir=$1
fs_tgt_dir=$2
model=$3
max_image_per_ltr=$4

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
set -o pipefail

stage=4
stop_stage=100

# CPC configs
n_predicts=3
n_cpc_clus=26
n_neg=32
feat_name=resnet34
echo ${feat_name}
if [ ${feat_name} = "vgg19" ]; then
    feat_dim=4096
elif [ ${feat_name} = "resnet152" ]; then
    feat_dim=2048
else
    feat_dim=512 
fi

# Dataset configs
tgt_dir=$(realpath $tgt_dir)
splits="train valid test"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # create audios without silence
    # get manifest
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py /home/hertin/data/LibriSpeech/dev-clean --ext flac --dest ${tgt_dir} --valid-percent 0
    mv $(pwd)/${tgt_dir}/train.tsv $(pwd)/${tgt_dir}/dev.tsv
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py /home/hertin/data/LibriSpeech/test-clean --ext flac --dest ${tgt_dir} --valid-percent 0
    mv $(pwd)/${tgt_dir}/train.tsv $(pwd)/${tgt_dir}/test.tsv
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py /home/hertin/data/LibriSpeech/train-clean-100 --ext flac --dest ${tgt_dir} --valid-percent 0
    # vad and  create audios without silence
    splits=(train test dev)
    for split in ${splits[@]}; do
        echo vad for $split
        python scripts/vads.py -r $RVAD_ROOT < ${tgt_dir}/${split}.tsv > ${tgt_dir}/${split}.vads
        python scripts/remove_silence.py --tsv ${tgt_dir}/${split}.tsv --vads ${tgt_dir}/${split}.vads --out ${tgt_dir}/no_silence
    done
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py ${tgt_dir}/no_silence --ext flac --dest ${tgt_dir}_ns --valid-percent 0.01
    head -n 150 ${tgt_dir}_ns/valid.tsv > ${tgt_dir}_ns/dev.tsv
    head -n 1 ${tgt_dir}_ns/valid.tsv > ${tgt_dir}_ns/test.tsv
    tail -n +150 ${tgt_dir}_ns/valid.tsv >> ${tgt_dir}_ns/test.tsv
fi

# Extract CPC features
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then 
    conda activate /home/lwang114/.conda/envs/cpc37
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
    for x in train train_100; do
        echo "Extract CPC features for ${x} set"
        python ${CPC_ROOT}/cpc/build_CPC_features.py \
            ${ckpt_path} \
            ${tgt_dir}/without_silence \
            ${ch2img_path} \
            manifest/asl_alphabet/${feat_name}/train \
            ${tgt_dir}/fs_feat/cpc_npredicts${n_predicts}_${n_neg}negatives_${max_image_per_ltr}images_per_ltr \
            --split ${x} --cpu
    done
    conda activate /home/hertin/.conda/envs/wav2vec 
fi

# Cluster CPC features
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then 
    tgt_cpc_dir=${tgt_dir}/fs_feat/cpc_npredicts${n_predicts}_${n_neg}negatives_${max_image_per_ltr}images_per_ltr
    #python scripts/image_cluster_sklearn.py \
    #    ${tgt_cpc_dir}/train_100.npy \
    #    --save-dir ${tgt_cpc_dir} \
    #    --n_clusters ${n_cpc_clus} --sample-pct 1.0 \
    #|| error "image_cluster_sklearn.py fails"

    for split in train; do
        python scripts/image_apply_cluster_sklearn.py \
            ${tgt_cpc_dir} --split ${split} --path ${tgt_cpc_dir}/CLUS${n_cpc_clus} \
        || error "image_apply_cluster_sklearn.py fails"

        echo "Compute NMI for ${split} set ..."
        python scripts/nmi.py \
            ${tgt_cpc_dir}/CLUS${n_cpc_clus}/${split}.src ${tgt_dir}/without_silence/${split}.trn || error "nmi.py failed"
   done
fi
