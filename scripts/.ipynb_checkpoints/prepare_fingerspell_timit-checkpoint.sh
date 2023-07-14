#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
timit_root=$1  # assume it is the upper-cased version
tgt_dir=$2
fs_tgt_dir=$3
model=$4
time_shift=$5

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

tgt_dir=$(realpath $tgt_dir)
setups="matched unmatched"
splits="test valid train train_text"

sph2wav=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
wav_dir=$tgt_dir/wav

n_clus=$(basename ${fs_tgt_dir})
fs_tgt_parent=$(dirname ${fs_tgt_dir})
model_name=$(basename ${fs_tgt_parent})
echo ${model_name}
if [ ${model_name} = "vgg19" ]; then
    feat_dim=4096
elif [ ${model_name} = "resnet152" ]; then
    feat_dim=2048
else
    feat_dim=512 
fi

stage=8
stop_stage=10
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  mkdir -p $tgt_dir $wav_dir
  find $timit_root/{TRAIN,TEST} -iname "*.WAV" > $tgt_dir/all_sph.flist
  cat $tgt_dir/all_sph.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).WAV#\1_\2#g' > $tgt_dir/all.uid
  paste -d' ' $tgt_dir/{all_sph.flist,all.uid} | \
    awk -v sph2wav=$sph2wav -v wav_dir=$wav_dir '{print sph2wav " -f wav " $1 " > " wav_dir "/" $2 ".wav"}' \
    > $tgt_dir/sph2wav.sh
  bash $tgt_dir/sph2wav.sh

  cat $tgt_dir/all.uid | awk -v wav_dir=$wav_dir '{print $1" "wav_dir"/"$1".wav"}' | sort > $tgt_dir/all_wav.scp

  echo stage 1.1
  cut -d' ' -f2 $tgt_dir/all_wav.scp | xargs -I{} soxi -s {} > $tgt_dir/all.dur
  echo stage 1.2
  paste -d' ' $tgt_dir/{all_wav.scp,all.dur} > $tgt_dir/all_wav_dur.scp
  rm $tgt_dir/{all.uid,all_sph.flist,sph2wav.sh}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  find $timit_root/{TRAIN,TEST} -iname "*.WRD" > $tgt_dir/all_wrd60.flist
  while read line; do
    if [ ! -f $line ]; then 
      >&2 echo "Cannot find transcription file '$line'" && exit 1;
    fi
    cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
  done < $tgt_dir/all_wrd60.flist > $tgt_dir/all.wrd60 

  cat $tgt_dir/all_wrd60.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).WRD#\1_\2#g' | \
    paste -d' ' - $tgt_dir/all.wrd60 | sort > $tgt_dir/all.wrd
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for s in $setups; do
    mkdir -p $tgt_dir/$s
    for x in $splits; do
      uid_path=config/timit_${s}/${x}.uid
      grep -w -f $uid_path $tgt_dir/all.wrd | cut -d' ' -f2- > $tgt_dir/$s/${x}_gt.wrd
    
      echo "/" > $tgt_dir/$s/$x.tsv &&  grep -w -f $uid_path $tgt_dir/all_wav_dur.scp | cut -d' ' -f2- | sed 's# #\t#'  >> $tgt_dir/$s/$x.tsv
    done
  done
  echo "done preparing unmatched and matched setups for TIMIT"
fi

# Create fingerspelling TIMIT cluster sequence
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  for s in ${setups}; do
    for x in $splits; do
      python scripts/create_fingerspell_dataset.py \
        --sp_path ${tgt_dir}/$s/${x}_gt.wrd \
        --fs_path ${fs_tgt_dir} \
        --out_path ${tgt_dir}/$s/${x}.phn || error "create_fingerspell_dataset.py fails" 
    done
  
    for x in $splits; do
      cat $tgt_dir/$s/${x}.phn
    done | tr ' ' '\n' | sort -u | awk '{print $1" "1}' > $tgt_dir/$s/dict.phn.txt
    #ln -sf $(realpath $tgt_dir/$s/dict.phn.txt) $tgt_dir/$s/dict.wrd.txt
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  for s in $setups; do
    echo prepare timit stage 5 $s
    # XXX zsh scripts/prepare_audio.sh $tgt_dir/$s $tgt_dir/$s/feat $model

    lm_dir=$tgt_dir/$s/phones
    fst_dir=$tgt_dir/$s/fst/phn_to_phn

    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/$s/train_text.phn --workers 10 --only-source --destdir $lm_dir --srcdict $tgt_dir/$s/dict.phn.txt
    $KENLM_ROOT/lmplz -o 3 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.03.arpa
    $KENLM_ROOT/build_binary $lm_dir/train_text_phn.03.arpa $lm_dir/train_text_phn.03.bin
    $KENLM_ROOT/lmplz -o 4 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.04.arpa
    $KENLM_ROOT/build_binary $lm_dir/train_text_phn.04.arpa $lm_dir/train_text_phn.04.bin
    
    python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir lm_arpa=$lm_dir/train_text_phn.03.arpa data_dir=$tgt_dir/$s in_labels=phn
  done
  echo "done preprocessing audio and text for wav2vec-U"
fi

# Prepare input features for APC training
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    cwd=$(pwd)
    cd ${APC_ROOT}
    for s in matched; do
        for x in train valid test; do
            python prepare_fingerspell.py \
                --fingerspell_dir ${fs_tgt_dir}/../ \
                --timit_dir ${tgt_dir}/${s} \
                --save_dir $(pwd)/data/fs_timit_${model_name}_${n_clus}/preprocessed \
                --split ${x} 
        done
    done
    cd ${cwd}
fi

# Train an APC model on fingerspelling TIMIT
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    cwd=$(pwd)
    cd ${APC_ROOT}
    python train_fingerspell_apc.py \
        --feature_dim ${feat_dim} \
        --learning_rate 0.001 \
        --data_path $(pwd)/data/fs_timit_${model_name}_${n_clus}/preprocessed \
        --time_shift ${time_shift} \
        --store_path $(pwd)/logs \
        --experiment_name fs_timit_${model_name}_time_shift_${time_shift} \
        || error "train_fingerspell_apc.py failed"
    cd ${cwd}
fi

# Extract APC features on fingerspelling TIMIT
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    cwd=$(pwd)
    cd ${APC_ROOT}
    for s in matched; do
        mkdir -p ${tgt_dir}/${s}/fs_feat/apc
        python extract_fingerspell_apc.py \
            --feature_dim ${feat_dim} \
            --time_shift ${time_shift} \
            --checkpoint_path $(pwd)/logs/fs_timit_${model_name}_time_shift_${time_shift}.dir/fs_timit_${model_name}_time_shift_${time_shift}__epoch_100.model \
            --data_path $(pwd)/data/fs_timit_${model_name}_${n_clus}/preprocessed/train \
            --save_order ${tgt_dir}/${s}/train.tsv \
            --save_prefix ${tgt_dir}/${s}/fs_feat/apc_time_shift_${time_shift}/train
    done
    cd ${cwd}
fi

# Cluster APC features
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    for s in matched; do
        n_apc_clus=29
        tgt_apc_dir=${tgt_dir}/${s}/fs_feat/apc_time_shift_${time_shift}/
        python scripts/image_cluster_faiss.py \
            ${tgt_apc_dir}/train.npy \
            --save-dir ${tgt_apc_dir} \
            -f "CLUS${n_apc_clus}" --sample-pct 1.0 \
            || error "image_cluster_faiss.py fails"

        python scripts/image_apply_cluster_faiss.py \
            ${tgt_apc_dir} --split train --path ${tgt_apc_dir}/CLUS${n_apc_clus} \
            || error "image_apply_cluster_faiss.py fails"
        
        python scripts/nmi.py \
            ${tgt_apc_dir}/CLUS${n_apc_clus}/train.src ${tgt_dir}/${s}/train.phn_fnames
    done
fi

# Linear separability of the APC features
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    cwd=$(pwd)
    cd ${APC_ROOT}
    for s in matched; do
        data_path=$(pwd)/data/fs_timit_${model_name}_${n_clus}/preprocessed
        label_path=${tgt_dir}/${s}
        for x in train valid; do
            cp ${label_path}/${x}.phn_fnames ${data_path}/${x}
        done
        exp_dir=$(pwd)/logs/fs_timit_${model_name}_time_shift_${time_shift}.dir
        python test_fingerspell_apc.py \
            --feature_dim ${feat_dim} \
            --time_shift ${time_shift} \
            --num_classes 26 \
            --data_path ${data_path} \
            --checkpoint_path ${exp_dir}/fs_timit_${model_name}_time_shift_${time_shift}__epoch_100.model \
            --experiment_path ${exp_dir} \
            --order_dir ${tgt_dir}/${s}
    done
    cd ${cwd}
fi
