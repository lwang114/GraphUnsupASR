#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -e
source_dir=$1
tgt_dir=$2
model=$3

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

mkdir -p $tgt_dir
echo stage 2
cp $source_dir/*.tsv $tgt_dir || true
cp $source_dir/*.wrd $tgt_dir || true
cp $source_dir/*.ltr $tgt_dir || true
cp $source_dir/*.phn $tgt_dir || true
cp $source_dir/dict* $tgt_dir || true

setopt shwordsplit
echo stage 3
for split in $all_splits; do
  echo stage 3 $split
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
  --save-dir $tgt_dir --checkpoint $model --layer $layer
done
echo stage 4
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py $tgt_dir/${train_split}.tsv \
--checkpoint $model --save-dir $tgt_dir -f "CLUS128" --sample-pct 0.5
echo stage 5
for split in $all_splits; do
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py $tgt_dir \
  --checkpoint $model --path $tgt_dir/CLUS128 --split $split
done
echo stage 6
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py $tgt_dir/${train_split}.npy --output $tgt_dir/pca --dim $dim
echo stage 7

for split in $all_splits; do
  echo stage 7 $split
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py $tgt_dir --split $split --save-dir $tgt_dir/precompute_pca$dim --pca-path $tgt_dir/pca/${dim}_pca --batch-size 1048000

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/precompute_pca$dim --cluster-dir $tgt_dir/CLUS128 \
  --split $split --save-dir $tgt_dir/precompute_pca${dim}_cls128_mean --pooling mean

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py $tgt_dir/precompute_pca${dim}_cls128_mean \
  --save-dir $tgt_dir/precompute_pca${dim}_cls128_mean_pooled --split $split
done

