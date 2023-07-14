import argparse
import numpy as np
import re
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred", 
        type=str,
        help="predicted label files (.src or .phn)",
    )
    parser.add_argument(
        "gold", 
        type=str,
        help="ground truth label files (.tsv or .wrd)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to save results",
        default=None,
    )
    parser.add_argument(
        "--fmt",
        default="faiss",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.pred, "r") as fp,\
        open(args.gold, "r") as fg:
        y_pred = []
        lines = fp.read().strip().split("\n")
        y_pred = [int(x) for l in lines for x in l.split()]
        sizes = [len(l.split()) for l in lines]
        y_pred = np.asarray(y_pred)

        y_gold = []
        vocab = dict()
        if args.gold.endswith("tsv"):
            lines = fg.read().strip().split("\n")
            _ = lines.pop(0)
            for line in lines:
                y_str = line.split("\t")[0].split("/")[-1]
                s, e = re.search(r"[A-Za-z1-9]+", y_str.split("_")[-1]).span()
                y_str = y_str[s:e]
                if not y_str in vocab:
                    vocab[y_str] = len(vocab)
                y = vocab[y_str]
                y_gold.append(y)
        elif args.gold.endswith("phn_fnames"):
            lines = fg.read().strip().split("\n")
            for line in lines:
                y_str = [
                    re.search(r"[\w]", fn.split("/")[-1]) 
                    for fn in line.strip().split()
                ]
                for y in y_str:
                    if not y in vocab:
                        vocab[y] = len(vocab)
                y_int = [vocab[y] for y in y_str]
                y_gold.extend(y_int)
        elif args.gold.endswith("trn") or args.gold.endswith("phn") or args.gold.endswith("wrd"):
            lines = fg.read().strip().split("\n")
            for line, size in zip(lines, sizes):
                y_str = [
                    c for c in line.strip().split() if c != "|"
                ]
                if not len(y_str):
                    y_str = "_"
                if not size == len(y_str) and args.fmt == "faiss":
                    print(f"Warning: pred label size {size} != gold label size {len(y_str)}")
                    if size < len(y_str):
                        y_str = y_str[:size]
                    elif size > len(y_str):
                        y_str.extend([y_str[-1]]*(size-len(y_str)))

                for y in y_str:
                    if not y in vocab:
                        vocab[y] = len(vocab)
                y_int = [vocab[y] for y in y_str]
                y_gold.extend(y_int)
        y_gold = np.asarray(y_gold)
        nmi = normalized_mutual_info_score(y_pred, y_gold)
        print(f"NMI: {nmi:.4f}", flush=True)
        purity = homogeneity_score(y_gold, y_pred)
        print(f"Homogeneity score: {purity:.4f}", flush=True)
        inv_purity = homogeneity_score(y_pred, y_gold)
        print(f"Completeness score: {inv_purity:.4f}", flush=True)
        if args.save_path:
            with open(args.save_path, "w") as f_out: 
                print(
                    f"NMI: {nmi:.4f}\n"\
                    f"Homogeneity score: {purity:.4f}\n"\
                    f"Completeness score: {inv_purity:.4f}", 
                    file=f_out,
                )

if __name__ == "__main__":
    main()
