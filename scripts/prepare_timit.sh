#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
timit_root=$1  # assume it is the upper-cased version
tgt_dir=$2
model=$3
orig_n_clus=$4

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
setups="unmatched"
splits="test valid train train_text"

sph2wav=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
wav_dir=$tgt_dir/wav

stage=1
stop_stage=6
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
  find $timit_root/{TRAIN,TEST} -iname "*.PHN" > $tgt_dir/all_phn60.flist
  while read line; do
    if [ ! -f $line ]; then 
      >&2 echo "Cannot find transcription file '$line'" && exit 1;
    fi
    cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
  done < $tgt_dir/all_phn60.flist > $tgt_dir/all.phn60 

  cat $tgt_dir/all_phn60.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).PHN#\1_\2#g' | \
    paste -d' ' - $tgt_dir/all.phn60 | \
    $KALDI_ROOT/egs/timit/s5/local/timit_norm_trans.pl -i - -m $KALDI_ROOT/egs/timit/s5/conf/phones.60-48-39.map -to 39 | \
    sort > $tgt_dir/all.phn

  python scripts/merge_repeated_phns.py $tgt_dir/all.phn $tgt_dir/all_merged.phn
  mv $tgt_dir/all_merged.phn $tgt_dir/all.phn
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for s in $setups; do
    mkdir -p $tgt_dir/$s
    for x in $splits; do
      uid_path=config/timit_${s}/${x}.uid
      grep -w -f $uid_path $tgt_dir/all.phn | cut -d' ' -f2- > $tgt_dir/$s/${x}.phn
      echo "/" > $tgt_dir/$s/$x.tsv &&  grep -w -f $uid_path $tgt_dir/all_wav_dur.scp | cut -d' ' -f2- | sed 's# #\t#'  >> $tgt_dir/$s/$x.tsv
    done
  done

  for s in $setups; do
    for x in $splits; do
      cat $tgt_dir/$s/${x}.phn
    done | tr ' ' '\n' | sort -u | awk '{print $1" "1}' > $tgt_dir/$s/dict.phn.txt
    ln -sf $(realpath $tgt_dir/$s/dict.phn.txt) $tgt_dir/$s/dict.wrd.txt
  done
  echo "done preparing unmatched and matched setups for TIMIT"
fi

echo "prepare timit stage 4: unsupervised segmentations for TIMIT"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  seg_model=unsup_seg_readout
  margin=10
  seg_dir=$tgt_dir/segmentations/phn_${seg_model}
  if [ $margin -gt 0 ]; then
    seg_dir=${seg_dir}_margin$margin
  fi

  for s in $setups; do
    out_dir=$tgt_dir/$s/phn_${seg_model}
    if [ $margin -gt 0 ]; then
      out_dir=${out_dir}
    fi
    echo $out_dir

    if [ ! -d $out_dir ]; then
      mkdir -p $out_dir
    fi
    
    for x in train valid test; do
      echo $seg_model,$x
      tsv_path=$tgt_dir/$s/$x.tsv
      python scripts/prepare_timit_${seg_model}.py \
        --in-dir $seg_dir \
        --out-path $out_dir/$x.src \
        --tsv-path $tsv_path
    done
  done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  for s in $setups; do
    echo prepare timit stage 5 $s
    zsh scripts/prepare_audio.sh $tgt_dir/$s $tgt_dir/$s/feat $model 512 14 $orig_n_clus
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for s in $setups; do
    echo prepare timit stage 6 $s
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
