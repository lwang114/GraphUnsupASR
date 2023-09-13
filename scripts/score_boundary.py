import argparse
import numpy as np
from pathlib import Path
import torch


def read_segments(fn):
    with open(fn, "r") as fp:
        segments = []
        for line in fp:
            clusts = list(map(int, line.strip().split()))
            clusts = torch.tensor(clusts)
            units, counts = clusts.unique_consecutive(return_counts=True)
            segments.append(counts[:-1].cumsum(0).numpy())
    return segments


def get_assignments(y, yhat, tolerance=1):
    matches = dict((i, []) for i in range(len(yhat)))
    for i, yhat_i in enumerate(yhat):
        dists = np.abs(y - yhat_i)
        idxs = np.argsort(dists)
        for idx in idxs:
            if dists[idx] <= tolerance:
                matches[i].append(idx)
    return matches


def get_counts(y, yhat):
    match_counter = 0
    dup_counter = 0
    miss_counter = 0
    used_idxs = []
    matches = get_assignments(y, yhat)
    dup_frames = []
    miss_frames = []

    for m, vs in matches.items():
        if len(vs) == 0:
            miss_frames.append(m)
            miss_counter += 1
            continue
        vs = sorted(vs)
        dup = False
        for v in vs:
            if v in used_idxs:
                dup = True
            else:
                dup = False
                used_idxs.append(v)
                match_counter += 1
                break
        if dup:
            dup_counter += 1
            dup_frames.append(m)

    return match_counter, dup_counter

 
def process_predictions(hypos, refs):
    pred_b_len = 0
    b_len = 0
    p_count = 0
    r_count = 0
    p_dup_count = 0
    r_dup_count = 0
    for yhat, y in zip(hypos, refs):
        b_len += len(y)
        pred_b_len += len(yhat)
        p, pd = get_counts(y, yhat)
        p_count += p
        p_dup_count += pd
        r, rd = get_counts(yhat, y)
        r_count += r
        r_dup_count += rd

    return p_count, p_dup_count, r_count, r_dup_count, pred_b_len, b_len


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0.0 else 0.0 

def rval(p, r):
    eps = 1e-8
    os = r / (p + eps) - 1                                                    
    r1 = np.sqrt((1 - r) ** 2 + os ** 2)                                                   
    r2 = (-os + r - 1) / (np.sqrt(2))                                                      
    rval = 1 - (np.abs(r1) + np.abs(r2)) / 2 
    return rval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file")
    parser.add_argument("--ref_file")
    parser.add_argument("--out_file")
    args = parser.parse_args()

    hypos = read_segments(args.hyp_file)
    refs = read_segments(args.ref_file)
    
    p_count, p_dup_count, r_count, r_dup_count, pred_b_len, b_len = process_predictions(hypos, refs)

    boundary_precision_harsh = 0.0
    boundary_recall_harsh = 0.0
    boundary_precision_lenient = 0.0
    boundary_recall_lenient = 0.0
    with open(args.out_file, "w") as f_out:
        if pred_b_len > 0:
            boundary_precision_harsh = p_count * 100.0 / pred_b_len
            boundary_precision_lenient = (p_count + p_dup_count) * 100.0 / pred_b_len
            info = f"Boundary precision (harsh): {boundary_precision_harsh}\n"\
                   f"Boundary precision (lenient): {boundary_precision_lenient}"
            print(info)
            print(info, file=f_out)

        if b_len > 0:
            boundary_recall_harsh = r_count * 100.0 / b_len
            boundary_recall_lenient = (r_count + r_dup_count) * 100.0 / b_len
            info = f"Boundary recall (harsh): {boundary_recall_harsh}\n"\
                   f"Boundary recall (lenient): {boundary_recall_lenient}"
            print(info)
            print(info, file=f_out)

        boundary_f1_harsh = f1(
            boundary_precision_harsh, boundary_recall_harsh,
        )
        boundary_f1_lenient = f1(
            boundary_precision_lenient, boundary_recall_lenient,
        )
        info = f"Boundary F1 (harsh): {boundary_f1_harsh}\n"\
               f"Boundary F1 (lenient): {boundary_f1_lenient}"
        print(info)
        print(info, file=f_out)

        boundary_rval_harsh = rval(
            boundary_precision_harsh, boundary_recall_harsh,
        )
        boundary_rval_lenient = rval(
            boundary_precision_lenient, boundary_recall_lenient,
        )
        info = f"Boundary Rval (harsh): {boundary_rval_harsh}\n"\
               f"Boundary Rval (lenient): {boundary_rval_lenient}"
        print(info)
        print(info, file=f_out)

if __name__ == "__main__":
    print(rval(0.882,0.764))
    print(rval(0.908,0.84))
    #main()
