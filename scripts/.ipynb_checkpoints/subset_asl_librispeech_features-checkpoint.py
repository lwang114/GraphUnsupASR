import argparse
import json
import numpy as np
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        help="Directory containing *.tsv and *.jsonlines of the whole data",
    )
    parser.add_argument(
        "--subset_path", 
        help="Directory containing *.tsv and *.jsonlines of the subset",
    )
    parser.add_argument(
        "--out_path",
        help="Directory containing the output features",)
    parser.add_argument("--split")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    in_path = Path(args.in_path)
    subset_path = Path(args.subset_path)
    out_path = Path(args.out_path)

    in_feats = np.load(in_path / f"{args.split}.npy")
    out_feats = []
    # TODO
    with open(subset_path / f"{args.split}.jsonlines", "r") as f_sub:
        for line in subset_path:
            utt = json.loads(line.rstrip("\n"))
            