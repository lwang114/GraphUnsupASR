#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
source_dir=$1
tgt_dir=$2
seg_dir=$3
stage=7
stop_stage=12

FAIRSEQ_ROOT=/home/hertin/workplace/wav2vec/fairseq
RVAD_ROOT=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised/rVADfast
echo satge 1
if [ -z "$4" ]
  then
    dim=512
  else
    dim=$4
fi

echo "using $dim dim for PCA"

if [ -z "$5" ]
  then
    layer=14
  else
    layer=$5
fi

n_clus=512

echo "extracting from layer $layer"

train_split=train
valid_split=valid
test_split=test

all_splits=($train_split)

if [[ -f "$source_dir/valid.tsv" ]]; then
    all_splits+=('valid')
fi

if [[ -f "$source_dir/test.tsv" ]]; then
    all_splits+=('test')
fi

echo "processing splits: $all_splits"

echo stage 7
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    for split in $all_splits; do
      echo stage 7 $split
      python scripts/merge_clusters.py $tgt_dir/precompute_pca$dim --cluster-dir $seg_dir \
         --split $split --save-dir $tgt_dir/precompute_pca${dim}_asru_seg_mean --pooling mean
    done
fi

echo stage 10
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    echo $tgt_dir
    python scripts/image_cluster_faiss.py $tgt_dir/precompute_pca512_asru_seg_mean/${train_split}.npy \
       --save-dir $tgt_dir/precompute_pca512_asru_seg_mean -f "CLUS$n_clus"
fi

echo stage 11
if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    for split in ${all_splits}; do
        python scripts/image_apply_cluster_faiss.py \
            $tgt_dir/precompute_pca512_asru_seg_mean \
            --path $tgt_dir/precompute_pca512_asru_seg_mean/CLUS$n_clus \
            --split $split
    done
fi

echo stage 12
if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    python scripts/extract_onehot_features.py \
        --in_dir $tgt_dir/precompute_pca512_asru_seg_mean/CLUS$n_clus/ \
        --out_dir $tgt_dir/precompute_pca512_asru_seg_mean_onehot_clus$n_clus \
        --suffix src \
        --fmt faiss \
        --save_int

    cp $tgt_dir/precompute_pca512_asru_seg_mean/*.tsv $tgt_dir/precompute_pca512_asru_seg_mean_onehot_clus$n_clus
    cp $tgt_dir/precompute_pca512_asru_seg_mean/*.phn $tgt_dir/precompute_pca512_asru_seg_mean_onehot_clus$n_clus

#    python scripts/extract_onehot_features.py \
#        --in_dir $tgt_dir/precompute_pca512_phn_gt_seg_mean/CLUS$n_clus/ \
#        --out_dir $tgt_dir/precompute_pca512_phn_gt_seg_mean_onehot_clus${n_clus}_float \
#        --suffix src \
#        --fmt faiss \
#        --save_int
#    cp $tgt_dir/precompute_pca512_phn_gt_seg_mean/*.tsv $tgt_dir/precompute_pca512_phn_gt_seg_mean_onehot_clus${n_clus}_float
#    cp $tgt_dir/precompute_pca512_phn_gt_seg_mean/*.phn $tgt_dir/precompute_pca512_phn_gt_seg_mean_onehot_clus${n_clus}_float

#    python scripts/extract_onehot_features.py \
#        --in_dir $tgt_dir/CLUS${orig_n_clus} \
#        --out_dir $tgt_dir/precompute_pca512_onehot_clus${n_clus}_float \
#        --suffix src \
#        --fmt faiss \

#    cp $tgt_dir/precompute_pca512/*.tsv $tgt_dir/precompute_pca512_onehot_clus${n_clus}_float
#    cp $tgt_dir/precompute_pca512/*.phn $tgt_dir/precompute_pca512_onehot_clus${n_clus}_float
#    python scripts/extract_onehot_features.py \
#        --in_dir $tgt_dir/precompute_pca512_asru_seg_mean/CLUS${n_clus} \
#        --out_dir $tgt_dir/precompute_pca512_asru_seg_mean_onehot_clus${n_clus} \
#        --suffix src \
#        --fmt faiss #\
#        --save_int
#    cp $tgt_dir/*.tsv $tgt_dir/precompute_pca512_asru_seg_mean_onehot_clus${n_clus}
#    cp $tgt_dir/*.phn $tgt_dir/precompute_pca512_asru_seg_mean_onehot_clus${n_clus}
fi
