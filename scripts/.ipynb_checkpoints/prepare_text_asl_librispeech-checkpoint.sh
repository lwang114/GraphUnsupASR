#!/usr/bin/env zsh

tgt_dir=$1
lm_dir=$tgt_dir

stage=0
stop_stage=100
# Create dict.txt, dict.wrd.txt and words.txt
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "prepare_text_asl_librispeech.sh: stage 1"
    cat $tgt_dir/train.wrd $tgt_dir/valid.wrd $tgt_dir/test.wrd > $tgt_dir/all.wrd
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/all.wrd --only-source --destdir $tgt_dir --thresholdsrc 0 --padding-factor 1 --dict-only
    cut -f1 -d' ' $tgt_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' >! $tgt_dir/words.txt
    cp $tgt_dir/dict.txt $tgt_dir/dict.wrd.txt
    echo "<SIL> 0" >> $tgt_dir/dict.wrd.txt
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then 
    echo "prepare_text_asl_librispeech.sh: stage 2"
    cp $tgt_dir/all.wrd $tgt_dir/lm.upper.lid.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $lm_dir/lm.upper.lid.txt --workers 70 --only-source --destdir $lm_dir --srcdict $lm_dir/dict.txt
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "prepare_text_asl_librispeech.sh: stage 3"
    paste $tgt_dir/words.txt $tgt_dir/words.txt >! $tgt_dir/lexicon.lst
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.0 --surround --lexicon $tgt_dir/lexicon.lst < $tgt_dir/lm.upper.lid.txt >! $lm_dir/lm.words.txt
    if [ ! -d $lm_dir/word_with_sil ]; then
        mkdir -p $lm_dir/word_with_sil
    fi
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 1.0 --lexicon $tgt_dir/$s/lexicon.lst < $tgt_dir/$s/lm.upper.lid.txt >! $lm_dir/word_with_sil/lm.words.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $lm_dir/word_with_sil/lm.words.txt --workers 70 --only-source --destdir $lm_dir/word_with_sil --srcdict $lm_dir/dict.txt
fi

# Create LMs
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "prepare_text_asl_librispeech.sh: stage 4"
    $KENLM_ROOT/lmplz -o 3 < $tgt_dir/lm.words.txt --discount_fallback --prune 0 0 0 >! $tgt_dir/kenlm.wrd.o3000.arpa
    $KENLM_ROOT/build_binary $tgt_dir/kenlm.wrd.o3000.arpa $tgt_dir/kenlm.wrd.o3000.bin
fi
