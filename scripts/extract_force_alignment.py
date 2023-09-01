import argparse
from collections import defaultdict
import json
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path")
    parser.add_argument("--align_path")
    parser.add_argument("--out_path")
    parser.add_argument("--split")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)                                       
    align_path = Path(args.align_path)                                             
    out_path = Path(args.out_path)

    with open(manifest_path / f"{args.split}.lengths", "r") as f_len:
        sizes = f_len.read().strip().split("\n")

    frames = []
    prev_sent_id = None
    idx = 0
    with open(align_path / "merged_alignment.txt", "r") as f_ali,\
        open(out_path / f"{args.split}.src", "w") as f_src: 
        for line in f_ali:
            sent_id, spk_id, start, dur, lbl = line.rstrip().split()
            if prev_sent_id is None:
                prev_sent_id = sent_id
            
            if sent_id != prev_sent_id:
                if len(frames) != sizes[idx]:
                    print(f"{len(frames)} != {sizes[idx]}")
                print(" ".join(frames), file=f_src)
                frames = []
                prev_sent_id = sent_id
                idx += 1
            
            start = int(float(start) * 50)
            end = int(float(end) * 50)
            frames.extend([lbl]*(end-start))

if __name__ == "__main__":
    main()
