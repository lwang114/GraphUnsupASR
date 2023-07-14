#!/bin/bash
#SBATCH --job-name="logs/timit_asru_graph"
#SBATCH --output="logs/%j.%N_timit_asru_graph.out"
#SBATCH --error="logs/%j.%N_timit_asru_graph.err"
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=2400
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=4
#SBATCH --threads-per-core=4
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:2
##SBATCH --mail-uer=lwang114@illinois.edu
##SBATCH --mail-type=ALL


source /home/lwang114/anaconda3/etc/profile.d/conda.sh
PYTHON_VIRTUAL_ENVIRONMENT=/home/lwang114/anaconda3/envs/fairseq
conda activate ${PYTHON_VIRTUAL_ENVIRONMENT}
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

export KALDI_ROOT=/ws/ifp-53_1/hasegawa/tools/kaldi 
export FAIRSEQ_ROOT=/ws/ifp-53_2/hasegawa/lwang114/spring2022/fairseq
export KENLM_ROOT=/ws/ifp-53_2/hasegawa/lwang114/spring2022/UnsupTTS/kenlm/build/bin
export RVAD_ROOT=/ws/ifp-53_2/hasegawa/lwang114/spring2022/UnsupTTS/rVADfast

TIMIT_DIR=/ws/ifp-53_2/hasegawa/lwang114/data/TIMIT
W2V=/home/hertin/models/wav2vec_vox_new.pt

set -eu
set -o pipefail

tgt_dir=$(pwd)/manifest/timit_norep  #unsup_seg
s=matched
if [ ! -d ${tgt_dir} ]; then
    mkdir -p $tgt_dir
fi

stage=4
stop_stage=4
echo stage 0, feature extraction
if [ $stage -ge 0 ] && [ $stop_stage -le 0 ]; then
    orig_n_clus=128
    bash scripts/prepare_timit.sh $TIMIT_DIR $tgt_dir $W2V $orig_n_clus
fi

echo stage 1, segmented ASR-U training
if [ $stage -ge 1 ] && [ $stop_stage -le 1 ]; then
    echo $tgt_dir
    PREFIX=w2v_unsup_gan_xp
    n_clus=512

    # For wav2vec-U, audio features are pre-segmented
    CONFIG_NAME=l1_w2vu_segmented_onehot_clus${n_clus}_skip6
    # TASK_DATA=$tgt_dir/$s/feat/precompute_pca512_asru_mean_onehot_clus$n_clus  
    TASK_DATA=$tgt_dir/$s/feat/precompute_pca512_asru_seg_mean_onehot_clus${n_clus}

    # Unpaired text input
    TEXT_DATA=$tgt_dir/$s/phones  # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=${tgt_dir}/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

    ckpt_dir=$(pwd)/multirun/2023-07-10/21-23-16
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/l1 \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.text_data=$TEXT_DATA \
        task.kenlm_path=$KENLM_PATH \
        common.user_dir=$(pwd)/wav2vecu_graph \
        model.code_penalty=0.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=0.0 'common.seed=range(0,1)' \
        checkpoint.save_dir='./' \
        hydra.run.dir=$ckpt_dir \
        hydra.sweep.dir=$ckpt_dir
fi

echo stage 2, pre-quantized ASR-U training
if [ $stage -ge 2 ] && [ $stop_stage -le 2 ]; then
    echo $tgt_dir
    PREFIX=w2v_unsup_gan_xp
    n_clus=128
    bsz=640  #160
    skip_size=6
    tri_size=2
    kernel_size=4

    # For wav2vec-U, audio features are pre-segmented
    #CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_skip${skip_size}_tri${tri_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool
    CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_skip${skip_size}_tri${tri_size}_bsz${bsz}_kernel${kernel_size}_softpool
    #CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_pos_skip${skip_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool
    #CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_skip${skip_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_decouple
    #CONFIG_NAME=l1_w2vu_skip6_bsz${bsz}_kernel4_posweight1_1
    TASK_DATA=$tgt_dir/$s/feat
    #TASK_DATA=$tgt_dir/$s/feat/precompute_pca512_onehot_clus${n_clus}_float
    #TASK_DATA=$tgt_dir/$s/feat/precompute_pca512
    #TASK_DATA=$tgt_dir/$s/feat/precompute_pca512_cls128_mean_onehot_clus${n_clus}_float

    # Unpaired text input
    TEXT_DATA=$tgt_dir/$s/phones  # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=${tgt_dir}/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
    #ckpt_dir=multirun/2023-06-22/18-43-33_3
    #ckpt_dir=$(pwd)/multirun/2023-06-24/16-07-17
    #ckpt_dir=$(pwd)/multirun/2023-07-07/22-23-20

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/l1 \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.text_data=$TEXT_DATA \
        task.kenlm_path=$KENLM_PATH \
        common.user_dir=$(pwd)/wav2vecu_graph \
        model.code_penalty=4.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=16.0 'common.seed=range(0,1)' #\
        #checkpoint.save_dir='./' \
        #hydra.run.dir=$ckpt_dir \
        #hydra.sweep.dir=$ckpt_dir
fi

echo stage 3, E2E ASR-U training
if [ $stage -ge 3 ] && [ $stop_stage -le 3 ]; then
    echo $tgt_dir
    PREFIX=w2v_unsup_gan_xp
    n_clus=512
    bsz=640
    skip_size=6
    kernel_size=1

    # For wav2vec-U, audio features are pre-segmented
    #CONFIG_NAME=l1_w2vu_skip${skip_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool
    #CONFIG_NAME=l1_gumbel_w2vu_clus${n_clus}_skip${skip_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool
    #CONFIG_NAME=l1_vq_w2vu_clus${n_clus}_skip${skip_size}_bsz${bsz}_kernel${kernel_size}_decouple
    #CONFIG_NAME=l1_vq_w2vu_clus${n_clus}_skip${skip_size}_bsz${bsz}_kernel${kernel_size}
    #CONFIG_NAME=l1_vq_w2vu_clus${n_clus}_skip${skip_size}_bsz${bsz}_kernel${kernel_size}_sample_join
    CONFIG_NAME=l1_pretrained_vq_w2vu_clus${n_clus}_skip${skip_size}_bsz${bsz}_kernel${kernel_size}
    TASK_DATA=$tgt_dir/$s/feat
    #TASK_DATA=$tgt_dir/$s/feat/precompute_pca512

    # Unpaired text input
    TEXT_DATA=$tgt_dir/$s/phones  # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=${tgt_dir}/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
    #ckpt_dir=multirun/2023-06-24/11-28-41
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/l1_vq \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.text_data=$TEXT_DATA \
        task.kenlm_path=$KENLM_PATH \
        common.user_dir=$(pwd)/wav2vecu_graph \
        model.code_penalty=0 model.gradient_penalty=0.0 \
        model.smoothness_weight=0.0 'common.seed=range(0,1)' #\
    #    checkpoint.save_dir='./' \
    #    hydra.run.dir=$ckpt_dir \
    #    hydra.sweep.dir=$ckpt_dir
fi

echo stage 4, ASR-U evaluation
if [ $stage -ge 4 ] && [ $stop_stage -le 4 ]; then
    n_clus=512
    #ckpt_dir=$(pwd)/multirun/2023-07-03/21-32-13  
    #ckpt_dir=$(pwd)/multirun/2023-07-05/18-27-01
    #ckpt_dir=$(pwd)/multirun/2023-07-03/11-37-15
    #ckpt_dir=$(pwd)/multirun/2023-06-24/16-07-17
    #ckpt_dir=$(pwd)/multirun/2023-07-08/16-04-52
    #ckpt_dir=$(pwd)/multirun/2023-07-08/23-04-22
    ckpt_dir=$(pwd)/multirun/2023-07-10/21-23-16

    #TASK_DATA=$tgt_dir/$s/feat
    TASK_DATA=$tgt_dir/$s/feat/precompute_pca512_asru_seg_mean_onehot_clus$n_clus
 
    HYDRA_FULL_ERROR=1 python w2vu_segmented_generate.py --config-dir $(pwd)/config/generate --config-name viterbi_segmented \
        fairseq.common.user_dir=$(pwd)/wav2vecu_graph \
        fairseq.task.data=$TASK_DATA \
        fairseq.task.text_data=$tgt_dir/$s/phones \
        fairseq.common_eval.path=$ckpt_dir/0/checkpoint_best.pt \
        fairseq.dataset.gen_subset=valid results_path=${ckpt_dir}/timit
fi
