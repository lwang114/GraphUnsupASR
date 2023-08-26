#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
source_dir=$1
tgt_dir=$2
model=$3
stage=1
stop_stage=9

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

if [ -z "$6" ]
  then
    orig_n_clus=128
  else
    orig_n_clus=$6
fi
n_clus=512

echo "extracting from layer $layer"

train_split=train
valid_split=valid
test_split=test

all_splits=($train_split)
echo $source_dir
if [[ -f "$source_dir/valid.tsv" ]]; then
    all_splits+=('valid')
fi

if [[ -f "$source_dir/test.tsv" ]]; then
    all_splits+=('test')
fi

echo "processing splits: $all_splits"

mkdir -p $tgt_dir
echo stage 2
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo $source_dir,$tgt_dir
    cp $source_dir/*.tsv $tgt_dir || true
    cp $source_dir/*.wrd $tgt_dir || true
    cp $source_dir/*.ltr $tgt_dir || true
    cp $source_dir/*.phn $tgt_dir || true
    cp $source_dir/dict* $tgt_dir || true
fi

setopt shwordsplit
echo stage 3
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    for split in $all_splits; do
      echo stage 3 $split
      python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
      --save-dir $tgt_dir --checkpoint $model --layer $layer
    done
fi

echo stage 4
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py $tgt_dir/${train_split}.tsv \
    --checkpoint $model --save-dir $tgt_dir -f "CLUS$orig_n_clus" --sample-pct 0.5
fi

echo stage 5
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    for split in $all_splits; do
      #python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py $tgt_dir \
      #--checkpoint $model --path $tgt_dir/CLUS$orig_n_clus --split $split
      python scripts/image_apply_cluster_faiss.py $tgt_dir --split $split --path $tgt_dir/CLUS$orig_n_clus
    done
fi

echo stage 7
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py $tgt_dir/${train_split}.npy --output $tgt_dir/pca --dim $dim
    for split in $all_splits; do
        echo stage 7 $split
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py $tgt_dir --split $split --save-dir $tgt_dir/precompute_pca$dim --pca-path $tgt_dir/pca/${dim}_pca --batch-size 1048000
    done
fi

echo stage 8
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    python scripts/extract_onehot_features.py \
        --in_dir $tgt_dir/CLUS${orig_n_clus} \
        --out_dir $tgt_dir/onehot_clus${orig_n_clus}_float \
        --suffix src \
        --fmt faiss

    for split in ${all_splits}; do
        cp $tgt_dir/onehot_clus${orig_n_clus}_float/$split.npy $tgt_dir/${split}_clus.npy
    done
fi

echo stage 9
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    m=10
    for x in $all_splits; do
        cp $tgt_dir/../phn_unsup_seg_readout/${x}.src $tgt_dir/$x.src
        cp $tgt_dir/../phn_gt_seg/${x}_gt.src $tgt_dir/${x}_gt.src
        cp $tgt_dir/../phn_gt_seg/$x.phn $tgt_dir/$x.phn
        cp $tgt_dir/../phn_gt_seg/$x.phn $tgt_dir/../$x.phn
    done
fi

# echo stage 13
# if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
#     for split in train valid
#         do
#         python scripts/align_segments.py \
#             --ref_file $tgt_dir/phn_gt_seg/$split.src \
#             --hyp_file $tgt_dir/CLUS$orig_n_clus/$split.src \
#             --out_file $tgt_dir/phn_gt_seg_aligned_clus${orig_n_clus}/$split.src
#     done
#
#    for split in train valid
#        do
#        python scripts/align_segments.py \
#            --ref_file $tgt_dir/phn_unsup_seg/$split.src \
#            --hyp_file $tgt_dir/CLUS$orig_n_clus/$split.src \
#            --out_file $tgt_dir/phn_unsup_seg_aligned_clus${orig_n_clus}/$split.src
#    done
# fi
