#!/usr/bin/env zsh

tgt_dir=$1
s=$2
lm_dir=$tgt_dir/$s/phones
fst_dir=$tgt_dir/$s/fst

stage=0
stop_stage=4
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then    
    cat $tgt_dir/$s/train.wrd $tgt_dir/$s/valid.wrd $tgt_dir/$s/test.wrd > $tgt_dir/$s/all.wrd
    cat $tgt_dir/$s/train.phn $tgt_dir/$s/valid.phn $tgt_dir/$s/test.phn > $tgt_dir/$s/all.phn
    
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/$s/all.wrd --only-source --destdir $tgt_dir/$s --thresholdsrc 0 --padding-factor 1 --dict-only
    cut -f1 -d' ' $tgt_dir/$s/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' >! $tgt_dir/$s/words.txt
fi

# Create dict.txt, dict.phn.txt and words.txt
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then    
    cp $tgt_dir/$s/all.wrd $tgt_dir/$s/lm.upper.lid.txt
    python scripts/g2p_wrd_to_phn.py \
        --wrd_path $tgt_dir/$s/all.wrd \
        --in_path $tgt_dir/$s/words.txt \
        --out_path $tgt_dir/$s/phones.txt \
        --comma_separated

    for split in train valid; do
        for suffix in phn wrd; do 
            echo $tgt_dir/$s/${split}.${suffix}
            cp $tgt_dir/$s/${split}.${suffix} $tgt_dir/$s/feat
            cp $tgt_dir/$s/${split}.${suffix} $tgt_dir/$s/feat/precompute_pca512
            cp $tgt_dir/$s/${split}.${suffix} $tgt_dir/$s/feat/precompute_pca512_cls128_mean
            cp $tgt_dir/$s/${split}.${suffix} $tgt_dir/$s/feat/precompute_pca512_cls128_mean_pooled
        done
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then    
    paste $tgt_dir/$s/words.txt $tgt_dir/$s/phones.txt >! $tgt_dir/$s/lexicon.lst

    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/$s/phones.txt --only-source --destdir $lm_dir --thresholdsrc 1 --padding-factor 1 --dict-only

    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $tgt_dir/$s/phones/dict.txt < $tgt_dir/$s/lexicon.lst >! $tgt_dir/$s/lexicon_filtered.lst
    # python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.25 --surround --lexicon $tgt_dir/$s/lexicon_filtered.lst < $tgt_dir/$s/lm.upper.lid.txt >! $lm_dir/lm.phones.filtered.txt
    python scripts/phonemize_with_sil.py $tgt_dir/$s/all.phn $tgt_dir/../ljspeech/without_silence/phones/lm.phones.filtered.txt $lm_dir/lm.phones.filtered.txt  # XXX For debugging only 
    cp $lm_dir/dict.txt $lm_dir/dict.phn.txt
    echo "<SIL> 0" >> $lm_dir/dict.phn.txt
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $lm_dir/lm.phones.filtered.txt --workers 70 --only-source --destdir $lm_dir --srcdict $lm_dir/dict.phn.txt
fi

# Create LMs
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    $KENLM_ROOT/lmplz -o 4 < $tgt_dir/$s/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 >! $tgt_dir/$s/kenlm.wrd.o40003.arpa
    $KENLM_ROOT/build_binary $tgt_dir/$s/kenlm.wrd.o40003.arpa $tgt_dir/$s/kenlm.wrd.o40003.bin

    python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir/phn_to_words_sil lm_arpa=$tgt_dir/$s/kenlm.wrd.o40003.bin data_dir=$lm_dir in_labels=phn "blank_symbol='<SIL>'"
    python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir/phn_to_words lm_arpa=$tgt_dir/$s/kenlm.wrd.o40003.bin wav2letter_lexicon=$tgt_dir/$s/lexicon_filtered.lst data_dir=$lm_dir in_labels=phn

    $KENLM_ROOT/lmplz -o 4 < $lm_dir/lm.phones.filtered.txt --discount_fallback >$lm_dir/lm.phones.filtered.04.arpa
    $KENLM_ROOT/build_binary $lm_dir/lm.phones.filtered.04.arpa $lm_dir/lm.phones.filtered.04.bin
    $KENLM_ROOT/lmplz -o 6 < $lm_dir/lm.phones.filtered.txt --discount_fallback >! $lm_dir/lm.phones.filtered.06.arpa
    $KENLM_ROOT/build_binary $lm_dir/lm.phones.filtered.06.arpa $lm_dir/lm.phones.filtered.06.bin
  
    python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir/phn_to_phn_sil lm_arpa=$lm_dir/lm.phones.filtered.06.arpa data_dir=$lm_dir in_labels=phn "blank_symbol='<SIL>'"
fi
