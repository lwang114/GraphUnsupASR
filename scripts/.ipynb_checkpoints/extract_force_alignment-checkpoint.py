import argparse
import json
import os
from pathlib import Path
from textgrids import TextGrid

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir") 
    parser.add_argument("--align_dir") 
    parser.add_argument("--out_dir") 
    parser.add_argument("--split")
    parser.add_argument(
        "--label_type", default="phone", choices={"phone", "word"},
    )
    return parser

def is_inside(x1, x2):
    if x1[0] >= x2[1] or x1[1] <= x2[0]:
        return False
    return True

def merge(y_str, y_int):
    y_int_merged = [y_int[0]]
    y_str_merged = [y_str[0]]
    for y_int_i, y_str_i  in zip(y_int[1:], y_str[1:]):
        if y_int_i != y_int_merged[-1]:
            y_int_merged.append(y_int_i)
            y_str_merged.append(y_str_i)
    return y_str_merged, y_int_merged

def read_librispeech_phone_alignment(phn_file):
    starts = []
    ends = []
    labels = []
    try:
        tg = TextGrid(phn_file)
    except:
        raise RuntimeError(f"Failed to open {phn_file}") 
    for phn in tg["phones"]:
        starts.append(phn.xmin)
        ends.append(phn.xmax)
        labels.append(''.join([x for x in phn.text if x.isalpha()]))
    return starts, ends, labels

def map_phones_to_words(
        phn_starts, phn_ends, phn_labels,
        wrd_starts, wrd_ends, wrd_labels,
    ):
    mapped_starts = [[] for _ in range(len(wrd_starts))]
    mapped_ends = [[] for _ in range(len(wrd_starts))]
    mapped_labels = [[] for _ in range(len(wrd_starts))]
    for p_s, p_e, p_lbl in zip(phn_starts, phn_ends, phn_labels):
        for w_idx, (w_s, w_e, w_lbl) in enumerate(
            zip(wrd_starts, wrd_ends, wrd_labels)
        ):
            if is_inside((p_s, p_e), (w_s, w_e)):
                mapped_starts[w_idx].append(p_s)
                mapped_ends[w_idx].append(p_e)
                mapped_labels[w_idx].append(p_lbl)
    # print(wrd_starts, wrd_ends, wrd_labels)
    # print(mapped_starts, mapped_ends, mapped_labels)
    return mapped_starts, mapped_ends, mapped_labels

def extract_librispeech_phone_alignments(
        manifest_dir, 
        align_dir, 
        out_dir, 
        frame_rate=0.02, 
        sr=16e3,
    ):
    manifest_dir = Path(manifest_dir)
    align_dir = Path(align_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    split_dirs = [s for s in align_dir.iterdir() if s.is_dir()]
    vocab = ['sil']
    for split in ['train', 'dev', 'test']:
        with open(manifest_dir / f"{split}.jsonlines", "r") as f_wrd_ali,\
            open(manifest_dir / f"feat/{split}.lengths", "r") as f_len,\
            open(manifest_dir / f"{split}.phn", "w") as f_phn,\
            open(out_dir / f"{split}_with_phn.jsonlines", "w") as f_phn_ali,\
            open(out_dir / f"{split}.src", "w") as f_src,\
            open(out_dir / f"{split}.src.txt", "w") as f_src_txt:
            sizes = list(map(int, f_len.read().strip().split("\n")))
            for line, size in zip(f_wrd_ali, sizes):
                utt = json.loads(line.rstrip("\n"))
                utt_id = utt["utterance_id"]
                wrd_starts = utt["begins"]
                wrd_ends = utt["ends"]
                wrd_labels = utt["words"]
                parts = utt_id.split("-")
                align_subdir = "/".join(parts[:-1])
                align_name = f"{utt_id}.TextGrid"
                found = False
                for split_dir in split_dirs:
                    align_path = split_dir / align_subdir / align_name
                    if align_path.exists():
                        found = True
                        break
                if not found:
                    raise ValueError(f"{align_path} not found")

                phn_starts, phn_ends, phn_labels = read_librispeech_phone_alignment(align_path)

                mapped_starts, mapped_ends, mapped_labels = map_phones_to_words(
                    phn_starts, phn_ends, phn_labels,
                    wrd_starts, wrd_ends, wrd_labels,
                )
                utt["phones"] = mapped_labels
                utt["phone_starts"] = mapped_starts
                utt["phone_ends"] = mapped_ends

                # Create phone-level transcript and frame-level phone sequence
                mapped_sequence = []
                mapped_int_sequence = []
                mapped_frame_sequence = []
                prev_phn_idx = 0
                for starts, ends, labels in zip(mapped_starts, mapped_ends, mapped_labels):
                    for start, end, phn in zip(starts, ends, labels):
                        if not phn in vocab:
                            vocab.append(phn)    
                        mapped_sequence.append(phn)
                        s = int(start / frame_rate)
                        e = int(end / frame_rate)
                        mapped_frame_sequence.extend([phn]*(e-s))
                        phn_idx = vocab.index(phn)
                        # In case of the same phone appearing consecutively 
                        if phn_idx == prev_phn_idx:
                            phn_idx = -phn_idx
                        prev_phn_idx = phn_idx
                        mapped_int_sequence.extend([str(phn_idx)]*(e-s))
                # print(f"Estimated size: {len(mapped_int_sequence)}, real size: {size}")
                if size > len(mapped_int_sequence):
                    # print('size > len(mapped_int_seq), gap: ', size, len(mapped_int_sequence), size - len(mapped_int_sequence))
                    gap = size - len(mapped_int_sequence)
                    mapped_int_sequence.extend(
                        [mapped_int_sequence[-1]]*gap
                    )
                    mapped_frame_sequence.extend(
                        [mapped_frame_sequence[-1]]*gap
                    )                   
                elif size < len(mapped_int_sequence):
                    # print('size < len(mapped_int_seq), gap: ', size, len(mapped_int_sequence), mapped_int_sequence[size:])
                    mapped_int_sequence = mapped_int_sequence[:size]
                    mapped_frame_sequence = mapped_frame_sequence[:size]
                    mapped_sequence = merge(mapped_frame_sequence, mapped_int_sequence)[0]
                assert len(mapped_int_sequence) == size

                f_src_txt.write(" ".join(mapped_frame_sequence)+"\n")
                f_src.write(" ".join(mapped_int_sequence)+"\n")
                f_phn.write(" ".join(mapped_sequence)+"\n")
                f_phn_ali.write(json.dumps(utt)+"\n")

def extract_librispeech_word_alignments(
        manifest_dir, 
        out_dir, 
        frame_rate=0.02, 
        sr=16e3,
    ):
    manifest_dir = Path(manifest_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = ['sil']
    for split in ['train', 'dev', 'test']:
        with open(manifest_dir / f"{split}.jsonlines", "r") as f_wrd_ali,\
            open(manifest_dir / f"feat/{split}.lengths", "r") as f_len,\
            open(out_dir / f"{split}.src", "w") as f_src,\
            open(out_dir / f"{split}.src.txt", "w") as f_src_txt:
            sizes = list(map(int, f_len.read().strip().split("\n")))
            for line, size in zip(f_wrd_ali, sizes):
                utt = json.loads(line.rstrip("\n"))
                utt_id = utt["utterance_id"]
                starts = utt["begins"]
                ends = utt["ends"]
                labels = utt["words"]

                # Create phone-level transcript and frame-level phone sequence
                frame_int_sequence = []
                frame_sequence = []
                prev_wrd_idx = 0
                for start, end, label in zip(starts, ends, labels):
                    frame_sequence.append(label)
                    s = int(start / frame_rate)
                    e = int(end / frame_rate)
                    frame_sequence.extend([label]*(e-s))
                    if not label in vocab:
                        vocab.append(label)
                        
                    wrd_idx = vocab.index(label)
                    # In case of the same word appearing consecutively 
                    if wrd_idx == prev_wrd_idx:
                        wrd_idx = - wrd_idx
                    prev_wrd_idx = wrd_idx
                    frame_int_sequence.extend([str(wrd_idx)]*(e-s))
                # print(f"Estimated size: {len(mapped_int_sequence)}, real size: {size}")
                if size > len(frame_int_sequence):
                    gap = size - len(frame_int_sequence)
                    frame_int_sequence.extend(
                        [frame_int_sequence[-1]]*gap
                    )
                    frame_sequence.extend(
                        [frame_sequence[-1]]*gap
                    )
                elif size < len(frame_int_sequence):
                    frame_int_sequence = frame_int_sequence[:size]
                    frame_sequence = frame_sequence[:size]
                print(utt_id)
                assert len(frame_int_sequence) == size

                f_src_txt.write(" ".join(frame_sequence)+"\n")
                f_src.write(" ".join(frame_int_sequence)+"\n")  
    print(f"Number of word types: {len(vocab)}")

    
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.label_type == "phone":
        extract_librispeech_phone_alignments(
            args.manifest_dir, 
            args.align_dir, 
            args.out_dir, 
        )
    else:
        extract_librispeech_word_alignments(
            args.manifest_dir,
            args.out_dir,
        )
    
if __name__ == "__main__":
    main()
