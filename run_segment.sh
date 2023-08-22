#!/bin/bash

source /home/lwang114/anaconda3/etc/profile.d/conda.sh
conda activate /home/lwang114/anaconda3/envs/fairseq

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

tgt_dir=$(pwd)/manifest/timit_norep/without_silence

stage=5
stop_stage=5
echo "Stage 4, generate unsupervised phoneme segmentation using UnsupSeg"
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    conda activate /home/lwang114/anaconda3/envs/unsup_seg
    cwd=$(pwd)
    cd ../UnsupSeg
 
    # python main.py
    python predict_all.py
      
    seg_dir=runs/timit_unsup_seg
    out_dir=$tgt_dir/phn_unsup_seg
    if [ ! -d $out_dir ]; then
        mkdir -p $out_dir  
    fi 

    for x in train valid test; do
        tsv_path=$tgt_dir/feat/$x.tsv
        python ${cwd}/scripts/prepare_timit_unsup_seg.py \
            --in-dir $seg_dir \
            --out-path $out_dir/$x.src \
            --tsv-path $tsv_path
    done
    
    conda deactivate
    cd ${cwd}
fi

echo "Stage 5, generate unsupervised phoneme segmentation using Readout"
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    src_dir=$tgt_dir/phn_unsup_seg
    gt_src_dir=$tgt_dir/phn_gt_seg
    feat_dir=$tgt_dir/feat/precompute_pca512

    cp $src_dir/*.src $feat_dir
    cp $gt_src_dir/*_gt.src $feat_dir
    for suff in tsv phn wrd npy lengths; do 
        cp $feat_dir/valid.$suff $feat_dir/test.$suff
    done

    conda activate /home/lwang114/anaconda3/envs/fairseq
    cwd=$(pwd)
    cd self-supervised-phone-segmentation
    
    python run_extracted.py
    python predict.py

    seg_dir=outputs/timit_unsup_seg_readout
    out_dir=$tgt_dir/phn_unsup_seg_readout
    if [ ! -d $out_dir ]; then
        mkdir -p $out_dir  
    fi 

    for x in train test; do
        cp $seg_dir/${x}_margin10.src $seg_dir/$x.src
        tsv_path=$tgt_dir/feat/$x.tsv
        python ${cwd}/scripts/prepare_timit_unsup_seg_readout.py \
            --in-dir $seg_dir \
            --out-path $out_dir/$x.src \
            --tsv-path $tsv_path
    done
    cp $out_dir/test.src $out_dir/valid.src
 
    conda deactivate
    cd ${cwd}
fi
