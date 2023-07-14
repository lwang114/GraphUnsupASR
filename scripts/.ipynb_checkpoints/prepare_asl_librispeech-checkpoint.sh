#!/bin/bash

#source /home/hertin/.bashrc
audio_root=
align_root=
video_root=
tgt_dir=
w2v=
topk=
min_length=
n_sp_clus=
n_vid_clus=
segment_type=wrd

# CPC parameters
n_predicts=3
n_neg=32
get_encoded=true
get_input=false
vid_feat_name=vgg19
vid_pooling=mean

splits="dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500"
extract_video_feature=true

. parse_options.sh || exit 1;

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

stage=6
stop_stage=7
src_dir=$tgt_dir/../librispeech960
if [ ! -d $src_dir ]; then
    mkdir -p $src_dir
fi

tgt_cpc_dir=${tgt_dir}/asl_feat/${vid_feat_name}_cpc_npredicts${n_predicts}_${n_neg}negatives
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Create .tsv and .trn files for ASL LibriSpeech"
    for x in ${splits}; do
        $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py \
            $audio_root/$x \
            --valid-percent 0.0 \
            --dest $src_dir
        mv $src_dir/train.tsv $src_dir/$x.tsv
    done
    
    splits="dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500"
    for x in ${splits}; do
        $FAIRSEQ_ROOT/examples/wav2vec/libri_labels.py \
            $src_dir/$x.tsv \
            --output-dir $src_dir \
            --output-name $x
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Create .jsonlines and .wrd files for ASL LibriSpeech"
    for x in train dev test; do
        python scripts/prepare_asl_librispeech.py \
            --manifest_path $src_dir \
            --align_path $align_root \
            --video_path $video_root/.. \
            --split $x \
            --out_path $tgt_dir \
            --topk $topk \
            --min_length $min_length
    done

    for x in train dev test; do
        cp $tgt_dir/$x.trn $tgt_dir/$x.wrd
    done
    
    for suff in tsv wrd; do
        cp $tgt_dir/dev.$suff $tgt_dir/valid.$suff
    done 
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Segmented acoustic feature extraction"
    bash scripts/prepare_segmented_audio.sh $tgt_dir $align_root $w2v $topk $n_sp_clus $segment_type
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    if $extract_video_feature; then
        echo "Stage 5: Video CPC feature extraction"
        cwd=$(pwd)
        cd $CPC_ROOT/cpc
        bash run_video.swb \
            --n-predicts $n_predicts \
            --n-neg $n_neg \
            --vocab-size $topk \
            --n-vid-clus $n_vid_clus \
            --get-encoded $get_encoded \
            --get-input $get_input \
            --feat-name $vid_feat_name \
            --video-dir $video_root \
            --root-dir ${cwd} \
            --n-predicts $n_predicts \
            --pooling $vid_pooling
        cd ${cwd}
    fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    if $extract_video_feature; then 
        echo "Stage 6: Prepare video CPC feature cluster unit sequence"
        cp -r $tgt_dir/feat/onehot_clus$n_sp_clus $tgt_dir/feat/onehot_clus${n_sp_clus}_vid_cpc_clus${n_vid_clus}
        
        for x in train valid test; do
            cp $tgt_cpc_dir/CLUS$n_vid_clus/$x.src $tgt_cpc_dir/CLUS$n_vid_clus/$x.wrd
            cp $tgt_dir/feat/onehot_clus${n_sp_clus}_vid_cpc_clus${n_vid_clus}/$x.tsv $tgt_cpc_dir/CLUS$n_vid_clus
            cp $tgt_cpc_dir/CLUS$n_vid_clus/$x.wrd $tgt_dir/feat/onehot_clus${n_sp_clus}_vid_cpc_clus${n_vid_clus}
        done
    fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    if $extract_video_feature; then 
        echo "Stage 7: Train a language model for video CPC feature cluster unit sequence"
        zsh scripts/prepare_text_asl_librispeech.sh $tgt_cpc_dir/CLUS$n_vid_clus
    fi
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    echo "Stage 10: Prepare text data for ASL LibriSpeech"
    zsh scripts/prepare_text_asl_librispeech.sh $tgt_dir
fi
