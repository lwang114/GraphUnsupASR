#!/bin/bash
#SBATCH --job-name="wav2vecu_timit"
#SBATCH --output="wav2vecu_timit.%j.%N.out"
#SBATCH --error="wav2vecu_timit.%j.%N.err"
#SBATCH --partition=gpux1
#SBATCH --time=24
#SBATCH --mail-user=lwang114@illinois.edu
#SBATCH --mail-type=ALL

source /opt/miniconda3/etc/profile.d/conda.sh
PYTHON_VIRTUAL_ENVIRONMENT=/home/hertin/.conda/envs/wav2vec
conda activate ${PYTHON_VIRTUAL_ENVIRONMENT}

stage=50
stop_stage=50

. parse_options.sh || exit 1;

export KALDI_ROOT=/home/hertin/softwares/kaldi
export FAIRSEQ_ROOT=/home/hertin/workplace/wav2vec/fairseq
export KENLM_ROOT=/home/hertin/softwares/kenlm/build/bin
export RVAD_ROOT=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised/rVADfast

W2V=/home/hertin/models/wav2vec_vox_new.pt

set -e
set -u
set -o pipefail

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    # create audios without silence
    # get manifest
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py /home/hertin/data/LibriSpeech/dev-clean --ext flac --dest $(pwd)/manifest/librispeech100 --valid-percent 0
    mv $(pwd)/manifest/librispeech100/train.tsv $(pwd)/manifest/librispeech100/dev.tsv
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py /home/hertin/data/LibriSpeech/test-clean --ext flac --dest $(pwd)/manifest/librispeech100 --valid-percent 0
    mv $(pwd)/manifest/librispeech100/train.tsv $(pwd)/manifest/librispeech100/test.tsv
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py /home/hertin/data/LibriSpeech/train-clean-100 --ext flac --dest $(pwd)/manifest/librispeech100 --valid-percent 0
    # vad and  create audios without silence
    splits=(train test dev)
    for split in ${splits[@]}; do
        echo vad for $split
        python scripts/vads.py -r $RVAD_ROOT < $(pwd)/manifest/librispeech100/${split}.tsv > $(pwd)/manifest/librispeech100/${split}.vads
        python scripts/remove_silence.py --tsv $(pwd)/manifest/librispeech100/${split}.tsv --vads $(pwd)/manifest/librispeech100/${split}.vads --out $(pwd)/manifest/librispeech100/no_silence
    done
    python /home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/wav2vec_manifest.py manifest/librispeech100/no_silence --ext flac --dest manifest/librispeech100_ns --valid-percent 0.01
    head -n 150 manifest/librispeech100_ns/valid.tsv > manifest/librispeech100_ns/dev.tsv
    head -n 1 manifest/librispeech100_ns/valid.tsv > manifest/librispeech100_ns/test.tsv
    tail -n +150 manifest/librispeech100_ns/valid.tsv >> manifest/librispeech100_ns/test.tsv

fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
    output_dir=$(pwd)/manifest/librispeech_text
    manifest_dir=$(pwd)/manifest/librispeech100
    zsh scripts/prepare_audio.sh ${manifest_dir} ${output_dir} ${W2V} 512 14
fi


if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ]; then
    lang=en
    text_file=/home/hertin/models/librispeech-lm-norm.txt
    output_dir=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised/manifest/librispeech100_ns/text
    phonemizer=G2P
    lid_model=/home/hertin/models/lid.176.bin
    min_phones=0
    zsh scripts/prepare_text.sh ${lang} ${text_file} ${output_dir} ${min_phones} ${phonemizer} ${lid_model} sil_prob
fi

if [ ${stage} -le 40 ] && [ ${stop_stage} -ge 40 ]; then
    output_dir=$(pwd)/manifest/timit
    TIMIT_DIR=/home/hertin/data/timit/TIMIT
    bash scripts/prepare_timit.sh ${TIMIT_DIR} ${output_dir} ${W2V}
fi

if [ ${stage} -le 50 ] && [ ${stop_stage} -ge 50 ]; then
    echo Stage: 50
    PREFIX=w2v_unsup_gan_xp
    cwd=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised

    # For wav2vec-U, audio features are pre-segmented
    CONFIG_NAME=w2vu
    TASK_DATA=$cwd/manifest/timit/matched/feat/precompute_pca512_cls128_mean_pooled  #$(pwd)/manifest/librispeech100_ns/phonemized/precompute_pca512_cls128_mean_pooled

    # Unpaired text input
    TEXT_DATA=$cwd/manifest/timit/matched/phones # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=$cwd/manifest/timit/matched/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

    PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
            -m --config-dir config/gan \
            --config-name $CONFIG_NAME \
            task.data=${TASK_DATA} \
            task.text_data=${TEXT_DATA} \
            task.kenlm_path=${KENLM_PATH} \
            common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
            model.code_penalty=2,4 model.gradient_penalty=2.0,1.5 \
            model.smoothness_weight=0.5,1.0,0.75 'common.seed=range(0,1)'
fi

if [ ${stage} -le 60 ] && [ ${stop_stage} -ge 60 ]; then
    echo Stage: 60
    ckpt_dir=$(pwd)/multirun/2023-08-31/01-15-21
    cwd=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised
    # tgt_dir=$cwd/manifest/timit
    tgt_dir=$(pwd)/manifest/timit_norep
    TASK_DATA=$tgt_dir/matched/feat/precompute_pca512_cls128_mean_pooled

    for x in test train; do 
        HYDRA_FULL_ERROR=1 python $cwd/w2vu_generate.py --config-dir $cwd/config/generate --config-name viterbi \
           fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
           fairseq.task.data=$TASK_DATA \
           fairseq.task.text_data=$tgt_dir/$s/phones \
           fairseq.common_eval.path=$ckpt_dir/0/checkpoint_best.pt \
           fairseq.dataset.gen_subset=$x results_path=$ckpt_dir/st
    done  
fi

if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ]; then
    echo Stage: 61
    ckpt_dir=$(pwd)/multirun/2023-08-31/01-15-21
    cwd=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised
    # tgt_dir=$cwd/manifest/timit
    tgt_dir=$(pwd)/manifest/timit_norep

    if [ ! -d $tgt_dir/feat ]; then
        mkdir $tgt_dir/feat
        cp -r $tgt_dir/unmatched/feat/test* $tgt_dir/feat
    fi

    if [ ! -d $tgt_dir/feat/CLUS128 ]; then
        cp -r $tgt_dir/matched/feat/CLUS128 $tgt_dir/feat
    fi

    if [ ! -d $tgt_dir/feat/pca ]; then
        cp -r $tgt_dir/matched/feat/pca $tgt_dir/feat
    fi

    # perform PCA, clustering, merging segment and mean pooling
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py $tgt_dir/feat --split test --save-dir $tgt_dir/feat/precompute_pca512 --pca-path $tgt_dir/feat/pca/512_pca --batch-size 1048000
  
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py $tgt_dir/feat \
    --checkpoint $W2V --path $tgt_dir/feat/CLUS128 --split test
   
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/feat/precompute_pca512 --cluster-dir $tgt_dir/feat/CLUS128 \
    --split test --save-dir $tgt_dir/feat/precompute_pca512_cls128_mean --pooling mean

    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py $tgt_dir/feat/precompute_pca512_cls128_mean \
    --save-dir $tgt_dir/feat/precompute_pca512_cls128_mean_pooled --split test

