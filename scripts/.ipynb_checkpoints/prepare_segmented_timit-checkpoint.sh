#!/bin/bash

set -e
timit_root=$1
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

tgt_dir=$(realpath $tgt_dir)

stage=2
stop_stage=5
# Extract frame-level phone sequence
if [ ${stage} -le 1 ] && [ $stop_stage -ge 1 ]; then
    if [ ! -d $tgt_dir/feat/CLUS39 ]; then
        mkdir -p $tgt_dir/feat/CLUS39
    fi
    for x in train valid; do
        python scripts/timit_norm_trans.py -i $tgt_dir/feat/CLUS60/$x.src.txt -m $KALDI_ROOT/egs/timit/s5/conf/phones.60-48-39.map --to 39 -o $tgt_dir/feat/CLUS39/$x.src.txt
        python scripts/map_phone_to_int.py $tgt_dir/feat/CLUS39/$x.src.txt $tgt_dir/feat/CLUS39/$x.src
    done
fi

# Merge features based on phone segmentation
if [ ${stage} -le 2 ] && [ $stop_stage -ge 2 ]; then
    for x in train valid; do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/feat/precompute_pca512 --cluster-dir $tgt_dir/feat/CLUS39 \
  --split $x --save-dir $tgt_dir/feat/precompute_pca512_cls39_mean --pooling mean
    done
fi