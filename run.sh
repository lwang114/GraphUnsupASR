#!/bin/bash
#SBATCH -J wav2vecu_graph
#SBATCH -o logs/%j_wav2vecu_graph.out
#SBATCH -e logs/%j_wav2vecu_graph.err
#SBATCH --mail-user=limingw@mit.edu
#SBATCH --qos=sched_level_2
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

##SBATCH --job-name="logs/timit_asru_graph"
##SBATCH --output="logs/%j.%N_timit_asru_graph.out"
##SBATCH --error="logs/%j.%N_timit_asru_graph.err"
##SBATCH --partition=gpu
##SBATCH --mem-per-cpu=2400
##SBATCH --time=24:00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=32
##SBATCH --sockets-per-node=1
##SBATCH --cores-per-socket=4
##SBATCH --threads-per-core=4
##SBATCH --export=ALL
##SBATCH --gres=gpu:v100:2
##SBATCH --mail-uer=lwang114@illinois.edu
##SBATCH --mail-type=ALL


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

server="satori"
if [ $server = "ifp" ]; then
    source /home/lwang114/anaconda3/etc/profile.d/conda.sh
    PYTHON_VIRTUAL_ENVIRONMENT=/home/lwang114/anaconda3/envs/fairseq
    conda activate ${PYTHON_VIRTUAL_ENVIRONMENT}

    export KALDI_ROOT=/ws/ifp-53_1/hasegawa/tools/kaldi 
    export FAIRSEQ_ROOT=/ws/ifp-53_2/hasegawa/lwang114/spring2022/fairseq
    export KENLM_ROOT=/ws/ifp-53_2/hasegawa/lwang114/spring2022/UnsupTTS/kenlm/build/bin
    export RVAD_ROOT=/ws/ifp-53_2/hasegawa/lwang114/spring2022/UnsupTTS/rVADfast

    TIMIT_DIR=/ws/ifp-53_2/hasegawa/lwang114/data/TIMIT
    W2V=/home/hertin/models/wav2vec_vox_new.pt
elif [ $server = "hal" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    PYTHON_VIRTUAL_ENVIRONMENT=/home/hertin/.conda/envs/wav2vec
    conda activate ${PYTHON_VIRTUAL_ENVIRONMENT}

    export KALDI_ROOT=/home/hertin/softwares/kaldi
    export FAIRSEQ_ROOT=/home/hertin/workplace/wav2vec/fairseq
    export KENLM_ROOT=/home/hertin/softwares/kenlm/build/bin
    export RVAD_ROOT=/home/hertin/workplace/wav2vec/fairseq/examples/wav2vec/unsupervised/rVADfast

    TIMIT_DIR=/home/hertin/data/timit/TIMIT
    W2V=/home/hertin/models/wav2vec_vox_new.pt
elif [ $server = "satori" ]; then
    source /nobackup/users/junruin2/anaconda3/etc/profile.d/conda.sh
    conda activate /nobackup/users/junruin2/anaconda3/envs/exp_spring2022

    export KALDI_ROOT=/nobackup/users/junruin2/pykaldi_expspring2022/tools/kaldi
    export FAIRSEQ_ROOT=/nobackup/users/junruin2/fairseq_expspring2022
    export VAD_ROOT=/nobackup/users/junruin2/rVAD/rVADfast_py_2.0
    export KENLM_ROOT=/nobackup/users/junruin2/kenlm/build/bin    

    TIMIT_DIR=/home/hertin/data/timit/TIMIT
    W2V=/home/hertin/models/wav2vec_vox_new.pt
fi

set -eu
set -o pipefail

tgt_dir=$(pwd)/manifest/timit_norep
s=unmatched
if [ ! -d ${tgt_dir} ]; then
    mkdir -p $tgt_dir
fi

stage=0
stop_stage=0
echo stage 0, feature extraction
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    orig_n_clus=128
    bash scripts/prepare_timit.sh $TIMIT_DIR $tgt_dir $W2V $orig_n_clus
fi

echo stage 1, pre-quantized ASR-U training: first pass
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    PREFIX=w2v_unsup_gan_xp
    n_clus=128
    bsz=640
    skip_size=6
    tri_size=2
    kernel_size=4

    # For wav2vec-U, audio features are pre-segmented
    CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_skip${skip_size}_tri${tri_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool
    TASK_DATA=$tgt_dir/$s/feat
    SEGMENT_DATA=$tgt_dir/$s/phn_unsup_seg_readout

    # Unpaired text input
    TEXT_DATA=$tgt_dir/$s/phones  # path to fairseq-preprocessed GAN data (phones dir)
    KENLM_PATH=$tgt_dir/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

    ckpt_dir=$(pwd)/multirun/timit_iter1
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/l1 \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.text_data=$TEXT_DATA \
        task.segment_data=$SEGMENT_DATA \
        task.kenlm_path=$KENLM_PATH \
        common.user_dir=$(pwd)/wav2vecu_graph \
        model.code_penalty=0.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=16.0 'common.seed=range(0,1)' \
        checkpoint.save_dir='./' \
        hydra.run.dir=$ckpt_dir \
        hydra.sweep.dir=$ckpt_dir
fi

echo stage 2, pre-quantized ASR-U alignment: first pass
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    n_clus=512
    ckpt_dir=$(pwd)/multirun/timit_iter1

    TASK_DATA=$tgt_dir/$s/feat
    for x in test valid train; do
        HYDRA_FULL_ERROR=1 python w2vu_generate.py --config-dir $(pwd)/config/generate --config-name viterbi \
            fairseq.common.user_dir=$(pwd)/wav2vecu_graph \
            fairseq.task.data=$TASK_DATA \
            fairseq.task.text_data=$tgt_dir/$s/phones \
            fairseq.common_eval.path=$ckpt_dir/0/checkpoint_best.pt \
            fairseq.dataset.gen_subset=$x results_path=$tgt_dir/$s/phn_asru_seg_iter1
    done
fi

echo stage 3, pre-quantized ASR-U training: second pass
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    PREFIX=w2v_unsup_gan_xp
    n_clus=128
    bsz=640
    skip_size=6
    tri_size=2
    kernel_size=4

    # For wav2vec-U, audio features are pre-segmented 
    CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_skip${skip_size}_tri${tri_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool

    TASK_DATA=$tgt_dir/$s/feat

    # Unpaired text input
    TEXT_DATA=$tgt_dir/$s/phones  # path to fairseq-preprocessed GAN data (phones dir)
    SEGMENT_DATA=$tgt_dir/$s/phn_asru_seg_iter1
    KENLM_PATH=${tgt_dir}/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

    ckpt_dir=$(pwd)/multirun/timit_iter2
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/l1 \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.segment_data=$SEGMENT_DATA \
        task.text_data=$TEXT_DATA \
        task.kenlm_path=$KENLM_PATH \
        common.user_dir=$(pwd)/wav2vecu_graph \
        model.code_penalty=0.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=16.0 'common.seed=range(0,1)' \
        checkpoint.save_dir='./' \
        hydra.run.dir=$ckpt_dir \
        hydra.sweep.dir=$ckpt_dir
fi

echo stage 4, pre-quantized ASR-U alignment: second pass
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    ckpt_dir=$(pwd)/multirun/timit_iter2

    TASK_DATA=$tgt_dir/$s/feat 
    for x in test valid train; do
        HYDRA_FULL_ERROR=1 python w2vu_generate.py --config-dir $(pwd)/config/generate --config-name viterbi \
            fairseq.common.user_dir=$(pwd)/wav2vecu_graph \
            fairseq.task.data=$TASK_DATA \
            fairseq.task.text_data=$tgt_dir/$s/phones \
            fairseq.common_eval.path=$ckpt_dir/0/checkpoint_best.pt \
            fairseq.dataset.gen_subset=$x results_path=$tgt_dir/$s/phn_asru_seg_iter2 \
            margin=0.0
    done
fi

echo stage 5, pre-quantized ASR-U training: third pass
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    PREFIX=w2v_unsup_gan_xp
    n_clus=128
    bsz=640
    skip_size=6
    tri_size=2
    kernel_size=4

    CONFIG_NAME=l1_w2vu_onehot_clus${n_clus}_skip${skip_size}_tri${tri_size}_bsz${bsz}_kernel${kernel_size}_posweight1_1_softpool
    TASK_DATA=$tgt_dir/$s/feat

    # Unpaired text input
    TEXT_DATA=$tgt_dir/$s/phones  # path to fairseq-preprocessed GAN data (phones dir)
    SEGMENT_DATA=$tgt_dir/$s/phn_asru_seg_iter2
    KENLM_PATH=${tgt_dir}/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

    ckpt_dir=$(pwd)/multirun/timit_iter3
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
        -m --config-dir config/l1 \
        --config-name $CONFIG_NAME \
        task.data=$TASK_DATA \
        task.segment_data=$SEGMENT_DATA \
        task.text_data=$TEXT_DATA \
        task.kenlm_path=$KENLM_PATH \
        common.user_dir=$(pwd)/wav2vecu_graph \
        model.code_penalty=0.0 model.gradient_penalty=0.0 \
        model.smoothness_weight=16.0 'common.seed=range(0,1)' \
        checkpoint.save_dir='./' \
        hydra.run.dir=$ckpt_dir \
        hydra.sweep.dir=$ckpt_dir
fi

echo stage 6, segmented ASR-U preprocessing
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    seg_dir=$tgt_dir/$s/phn_asru_seg_iter2
    zsh scripts/prepare_segmented_audio.sh $TIMIT_DIR $tgt_dir $seg_dir
fi

echo stage 7, segmented ASR-U training
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
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

    ckpt_dir=$(pwd)/multirun/timit_segmented 
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

echo stage 8, segmented ASR-U evaluation
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    ckpt_dir=$(pwd)/multirun/timit_segmented

    TASK_DATA=$tgt_dir/$s/feat/precompute_pca512_asru_seg_mean_onehot_clus512
    for x in test valid train; do
        HYDRA_FULL_ERROR=1 python w2vu_segmented_generate.py --config-dir $(pwd)/config/generate --config-name viterbi_segmented \
            fairseq.common.user_dir=$(pwd)/wav2vecu_graph \
            fairseq.task.data=$TASK_DATA \
            fairseq.task.text_data=$tgt_dir/$s/phones \
            fairseq.common_eval.path=$ckpt_dir/0/checkpoint_best.pt \
            fairseq.dataset.gen_subset=$x results_path=$ckpt_dir
    done
fi

echo stage 9, Kaldi self-training
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    checkpoint_root=$(pwd)/multirun/timit_segmented
    
    LM_PATH=$tgt_dir/$s/phones/train_text_phn.04.arpa
    KENLM_PATH=$tgt_dir/$s/phones/train_text_phn.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
    
    cwd=$(pwd)
    cd kaldi_self_train/st
    # if [ -L utils ]; then
    #     rm utils
    # fi
    # if [ -L steps ]; then
    #     rm steps
    # fi 
    # ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
    # ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
    cp $tgt_dir/$s/phones/dict.phn.txt $tgt_dir/$s/feat/precompute_pca512

    if [ ! -d ${checkpoint_root}/st ]; then
        mkdir -p ${checkpoint_root}/st
        mv ${checkpoint_root}/*.* ${checkpoint_root}/st
    fi
    bash train.sh $tgt_dir/$s/feat/precompute_pca512 \
        ${checkpoint_root}/st \
        ${checkpoint_root}/st \
        ${LM_PATH} \
        ${KENLM_PATH}

    bash decode_phone.sh ${checkpoint_root}/st \
        7.0.0 tri2b \
        steps/decode.sh
    cd ${cwd}
fi
