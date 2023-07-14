import argparse
from collections import defaultdict
from pathlib import Path
import re


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        help="Path to the .src file",
    )
    parser.add_argument(
        "--lengths_path",
        type=str,
        help="Path to the .lengths file",
    )
    parser.add_argument(
        "--wrd_path",
        type=str,
        help="Path to the word boundary file",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        help="Prefix to the output features",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    src_path = Path(args.src_path)
    lengths_path = Path(args.lengths_path)

    if args.wrd_path:
        with open(src_path, "r") as fc,\
            open(lengths_path, "r") as fl,\
            open(args.wrd_path, "r") as fw,\
            open(args.out_prefix+".wrd", "w") as fo:
            cluster_idxs = fc.read().strip().split("\n")
            lengths = [int(l) for l in fl.read().strip().split("\n")]
            start = 0
            for line, l in zip(fw, lengths):
                words = line.strip().split()
                wb = []
                sent = []
                prev_end = start
                for w in words:
                    w = [c for c in w if c.isalpha()]
                    end = start + len(w)
                    sent.append(
                        ",".join(cluster_idxs[start:end])
                    )
                    start = end
                if not len(sent):
                    end += 1
                    start = end
                assert end - prev_end == l
                fo.write(" ".join(sent)+"\n")

    with open(src_path, "r") as fc,\
         open(lengths_path, "r") as fl,\
         open(args.out_prefix+".phn", "w") as fo:
        lengths = [int(l) for l in fl.read().strip().split("\n")]
        cluster_idxs = fc.read().strip().split("\n")
        start = 0
        for l in lengths:
            fo.write(
                " ".join(cluster_idxs[start:start+l])+"\n"
            )
            start += l

if __name__ == "__main__":
    main()


