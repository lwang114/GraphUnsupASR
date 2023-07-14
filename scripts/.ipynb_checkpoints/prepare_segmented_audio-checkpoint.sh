#!/bin/bash

tgt_dir=$1
align_dir=$2
w2v=$3
n_word=$4
n_clus=$5
segment_type=$6

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
splits="train dev test"
dim=512
n_phone=39
n_phn_clus=100

stage=0
stop_stage=100
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "prepare_segmented_audio.sh: stage 3"
    for x in $splits; do
        python scripts/wav2vec_extract_segment_features.py \
            $tgt_dir \
            --split $x \
            --save-dir $tgt_dir/feat \
            --checkpoint $w2v \
            --layer 14 \
            --reduction none
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "prepare_segmented_audio.sh: stage 4"
    python scripts/extract_force_alignment.py \
        --manifest_dir $tgt_dir \
        --align_dir $align_dir \
        --out_dir $tgt_dir/feat/CLUS${n_phone} \
        --label_type phone || error "extract_force_alignment.py failed"
    
    python scripts/extract_force_alignment.py \
        --manifest_dir $tgt_dir \
        --align_dir $align_dir \
        --out_dir $tgt_dir/feat/CLUS${n_word} \
        --label_type word || error "extract_force_alignment.py failed"
        
    for suff in phn wrd tsv; do
        cp $tgt_dir/*.$suff $tgt_dir/feat
    done
    
    for suff in phn wrd tsv npy lengths; do
        cp $tgt_dir/feat/dev.$suff $tgt_dir/feat/valid.$suff
    done
    
    cp $tgt_dir/feat/CLUS${n_phone}/dev.src $tgt_dir/feat/CLUS${n_phone}/valid.src
    cp $tgt_dir/feat/CLUS${n_phone}/dev.src.txt $tgt_dir/feat/CLUS${n_phone}/valid.src.txt
    cp $tgt_dir/feat/CLUS${n_word}/dev.src $tgt_dir/feat/CLUS${n_word}/valid.src
    cp $tgt_dir/feat/CLUS${n_word}/dev.src.txt $tgt_dir/feat/CLUS${n_word}/valid.src.txt
fi

# KMeans clustering
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "prepare_segmented_audio.sh: stage 5"
    python scripts/image_cluster_faiss.py $tgt_dir/feat/train.npy \
        --save-dir $tgt_dir/feat -f "CLUS128"
    for split in train valid; do
      python scripts/image_apply_cluster_faiss.py $tgt_dir/feat \
        --path $tgt_dir/feat/CLUS128 --split $split
    done
fi

# PCA
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "prepare_segmented_audio.sh: stage 6"
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py $tgt_dir/feat/train.npy --output $tgt_dir/feat/pca --dim $dim

    for split in train valid; do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py $tgt_dir/feat --split $split --save-dir $tgt_dir/feat/precompute_pca$dim --pca-path $tgt_dir/feat/pca/${dim}_pca --batch-size 1048000
    done
fi

# Merge clusters
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "prepare_segmented_audio.sh: stage 7"
    for split in train valid; do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/feat/precompute_pca$dim --cluster-dir $tgt_dir/feat/CLUS128 \
        --split $split --save-dir $tgt_dir/feat/precompute_pca${dim}_cls128_mean --pooling mean

        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py $tgt_dir/feat/precompute_pca${dim}_cls128_mean \
        --save-dir $tgt_dir/feat/precompute_pca${dim}_cls128_mean_pooled --split $split
    done
    
    for split in train valid; do
        for suff in tsv wrd; do
            cp $tgt_dir/$split.$suff $tgt_dir/feat/precompute_pca512
            cp $tgt_dir/$split.$suff $tgt_dir/feat/precompute_pca512_cls128_mean
            cp $tgt_dir/$split.$suff $tgt_dir/feat/precompute_pca512_cls128_mean_pooled
        done
    done

    for suff in npy lengths; do
        cp $tgt_dir/feat/precompute_pca512/dev.$suff $tgt_dir/feat/precompute_pca512/valid.$suff
        cp $tgt_dir/feat/precompute_pca512_cls128_mean/dev.$suff $tgt_dir/feat/precompute_pca512_cls128_mean/valid.$suff
        cp $tgt_dir/feat/precompute_pca512_cls128_mean_pooled/dev.$suff $tgt_dir/feat/precompute_pca512_cls128_mean_pooled/valid.$suff
    done
fi

# Merge clusters using ground truth phoneme boundaries
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "prepare_segmented_audio.sh: stage 8"
    for split in train valid; do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/feat/precompute_pca$dim --cluster-dir $tgt_dir/feat/CLUS${n_phone} \
        --split $split --save-dir $tgt_dir/feat/precompute_pca${dim}_cls${n_phone}_mean --pooling mean
    done
    
    python scripts/image_cluster_faiss.py $tgt_dir/feat/precompute_pca${dim}_cls${n_phone}_mean/train.npy \
        --save-dir $tgt_dir/feat/precompute_pca${dim}_cls${n_phone}_mean -f "CLUS${n_phn_clus}"
    for split in train valid; do
      python scripts/image_apply_cluster_faiss.py $tgt_dir/feat/precompute_pca${dim}_cls${n_phone}_mean \
        --path $tgt_dir/feat/precompute_pca${dim}_cls${n_phone}_mean/CLUS${n_phn_clus} --split $split
    done
    
    python scripts/nmi.py \
        $tgt_dir/feat/precompute_pca${dim}_cls${n_phone}_mean/CLUS${n_phn_clus}/train.src $tgt_dir/train.phn || error "nmi.py failed"
        
    for suff in tsv wrd phn; do
        cp $tgt_dir/*.$suff $tgt_dir/feat/precompute_pca512_cls${n_phone}_mean
    done
    
    for suff in tsv wrd phn; do
        cp $tgt_dir/feat/precompute_pca512_cls${n_phone}_mean/dev.$suff $tgt_dir/feat/precompute_pca512_cls${n_phone}_mean/valid.$suff
    done
fi

# Merge clusters using gold/predicted word boundaries
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    echo "prepare_segmented_audio.sh: stage 9"
    if [ $segment_type = "wrd" ]; then
        for split in train valid; do
            python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/feat/precompute_pca$dim --cluster-dir $tgt_dir/feat/CLUS${n_word} \
            --split $split --save-dir $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_mean --pooling mean 
        done

        python scripts/image_cluster_faiss.py $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_mean/train.npy \
            --save-dir $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_mean -f "CLUS${n_clus}"

        for split in train valid; do
          python scripts/image_apply_cluster_faiss.py $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_mean \
              --path $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_mean/CLUS${n_clus} --split $split
        done

        python scripts/nmi.py \
            $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_mean/CLUS${n_clus}/train.src $tgt_dir/train.wrd || error "nmi.py failed"

        for suff in tsv wrd phn; do
            cp $tgt_dir/*.$suff $tgt_dir/feat/precompute_pca512_cls${n_word}_mean
        done

        for suff in tsv wrd phn; do
            cp $tgt_dir/feat/precompute_pca512_cls${n_word}_mean/dev.$suff $tgt_dir/feat/precompute_pca512_cls${n_word}_mean/valid.$suff
        done
    elif [ $segment_type = "phn" ]; then
        for split in train valid; do
            python scripts/merge_clusters.py $tgt_dir/feat/precompute_pca${dim} --cluster-dir $tgt_dir/feat/CLUS${n_word}_given_${segment_type} \
            --split $split --save-dir $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_given_${segment_type}_mean --pooling mean \
            --fmt faiss
        done

        python scripts/image_cluster_faiss.py $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_given_${segment_type}_mean/train.npy \
            --save-dir $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_given_${segment_type}_mean -f "CLUS${n_clus}"

        for split in train valid; do
          python scripts/image_apply_cluster_faiss.py $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_given_${segment_type}_mean \
              --path $tgt_dir/feat/precompute_pca${dim}_cls${n_word}_given_${segment_type}_mean/CLUS${n_clus} --split $split
        done
        
        for suff in tsv wrd phn; do
            cp $tgt_dir/*.$suff $tgt_dir/feat/precompute_pca512_cls${n_word}_given_${segment_type}_mean
        done

        for suff in tsv wrd phn; do
            cp $tgt_dir/feat/precompute_pca512_cls${n_word}_given_${segment_type}_mean/dev.$suff $tgt_dir/feat/precompute_pca512_cls${n_word}_given_${segment_type}_mean/valid.$suff
        done
    else
        error "Unknown segment type $segment_type"
    fi
fi

# Extract one-hot word-level features
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    echo "prepare_segmented_audio.sh: stage 10"
    if [ $segment_type = "wrd" ]; then
        python scripts/extract_onehot_features.py \
            --in_dir manifest/asl_librispeech960_${n_word}words/feat/precompute_pca512_cls${n_word}_mean/CLUS${n_clus}/ \
            --out_dir manifest/asl_librispeech960_${n_word}words/feat/onehot_clus${n_clus} --suffix src --fmt sklearn

        for suff in tsv wrd phn; do
            cp $tgt_dir/*.$suff $tgt_dir/feat/onehot_clus${n_clus}
        done

        for suff in tsv wrd phn; do
            cp $tgt_dir/feat/onehot_clus${n_clus}/dev.$suff $tgt_dir/feat/onehot_clus${n_clus}/valid.$suff
        done
    elif [ $segment_type = "phn" ]; then
        python scripts/extract_onehot_features.py \
            --in_dir manifest/asl_librispeech960_${n_word}words/feat/precompute_pca512_cls${n_word}_given_${segment_type}_mean/CLUS${n_clus} \
            --out_dir manifest/asl_librispeech960_${n_word}words/feat/onehot_clus${n_clus}_given_$segment_type --suffix src --fmt sklearn

        for suff in tsv wrd phn; do
            cp $tgt_dir/*.$suff $tgt_dir/feat/onehot_clus${n_clus}_given_$segment_type
        done

        for suff in tsv wrd phn; do
            cp $tgt_dir/feat/onehot_clus${n_clus}_given_$segment_type/dev.$suff $tgt_dir/feat/onehot_clus${n_clus}_given_$segment_type/valid.$suff
        done
    else
        error "Unknown segment type $segment_type"
    fi
fi

# Extract one-hot phone-level and multi-hot word-level features
# if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
#     echo "prepare_segmented_audio.sh: stage 11"
#     pooling=mean
#     cls_dir=$tgt_dir/feat/CLUS${n_word}_given_phn_phn_level
#     python scripts/extract_onehot_features.py \
#         --in_dir manifest/asl_librispeech960_${n_word}words/feat/precompute_pca512_cls${n_phone}_mean/CLUS${n_phn_clus} \
#         --out_dir manifest/asl_librispeech960_${n_word}words/feat/onehot_clus${n_phn_clus}_phn --suffix src --fmt sklearn

#     for suff in tsv wrd phn; do
#         cp $tgt_dir/*.$suff $tgt_dir/feat/onehot_clus${n_phn_clus}_phn
#     done

#     for suff in tsv wrd phn; do
#         cp $tgt_dir/feat/onehot_clus${n_phn_clus}_phn/dev.$suff $tgt_dir/feat/onehot_clus${n_phn_clus}_phn/valid.$suff
#     done
    
#     python scripts/find_max_segment_length.py \
#         $tgt_dir/feat/onehot_clus${n_phn_clus}_phn \
#         --cluster-dir $cls_dir \
#         --fmt faiss
    
#     for split in valid train; do
#         python scripts/merge_clusters.py \
#             $tgt_dir/feat/onehot_clus${n_phn_clus}_phn \
#             --cluster-dir $cls_dir \
#             --split $split \
#             --save-dir $tgt_dir/feat/onehot_clus${n_phn_clus}_phn_$pooling \
#             --pooling $pooling \
#             --max_segment_length $cls_dir/max_segment_length.txt \
#             --fmt faiss
#     done
# fi