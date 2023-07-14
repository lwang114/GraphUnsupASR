import argparse
from dtw import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
import torch
from pathlib import Path
from shutil import copyfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file")
    parser.add_argument("--hyp_file")
    parser.add_argument("--out_file")
    return parser

def read_segments(in_path):
    with open(in_path, "r") as f_in:
        lines = f_in.read().strip().split("\n")
        all_segments = []
        all_units = []
        for l in lines:
            units = list(map(int, l.split()))
            units = torch.LongTensor(units)
            units, counts = torch.unique_consecutive(
                units,
                return_counts=True
            )
            offset = 0
            segments = []
            for c in counts:
                offset += c.item()
                segments.append(offset)
            all_units.append(units.data.tolist())
            all_segments.append(segments)
    return all_segments, all_units

def align(hyp, ref):
    alignment = dtw(hyp, ref, keep_internals=True)
    prev_index1 = -1 
    prev_index2 = -1
    index1s = []
    index2s = []
    hyp_units = []
    ref_units = []
    for index1, index2 in zip(alignment.index1s, alignment.index2s):
        if prev_index2 == index2:
            prev_index1 = index1
            continue
        if prev_index2 >= 0:
            index1s.append(prev_index1)
            index2s.append(prev_index2)
            hyp_units.append(hyp[prev_index1])
            ref_units.append(ref[prev_index2])
        prev_index1 = index1
        prev_index2 = index2
    index1s.append(prev_index1)
    index2s.append(prev_index2)
    hyp_units.append(hyp[prev_index1])
    ref_units.append(ref[prev_index2])
    return index1s, index2s, hyp_units, ref_units

def write_segments(labels, hyp_idxs, ref_idxs, f_out):
    offset = 0
    segments = []
    n_ref_seg = 0
    n_hyp_seg = 0
    for h_i, r_i in zip(hyp_idxs, ref_idxs):
        size = h_i - offset + 1 
        segments.extend([str(labels[r_i])]*size)
        offset += size
        n_ref_seg += 1
        if size > 0:
            n_hyp_seg += 1

    # print("n_ref_seg: ", n_ref_seg)
    # print("n_hyp_seg: ", n_hyp_seg)
    # print()
    print(' '.join(segments), file=f_out)
    return n_ref_seg, n_hyp_seg

def main():
    parser = get_parser()
    args = parser.parse_args()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ref_segments, ref_units = read_segments(args.ref_file)
    hyp_segments, hyp_units = read_segments(args.hyp_file)
    tot_ref, tot_hyp = 0, 0
    with open(out_file, 'w') as f_out:
        for ref_labels, hyp, ref in zip(ref_units, hyp_segments, ref_segments):
            index1s, index2s, new_hyp, new_ref = align(hyp, ref)
            # print("hyp_units: ", new_hyp)
            # print("ref_units: ", new_ref)
            n_ref, n_hyp = write_segments(ref_labels, index1s, index2s, f_out)
            tot_ref += n_ref
            tot_hyp += n_hyp
    print(f"Coverage: {100.0 * tot_hyp / tot_ref}")

if __name__ == "__main__":
    main() 
