import argparse
from shutil import copyfile
import numpy as np
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--suffix", default="wrd")
    parser.add_argument("--fmt", choices={"sklearn", "faiss"}, default="faiss")
    parser.add_argument("--save_int", action="store_true")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    vocab = []
    sents = {}
    for x in ["train", "valid"]:
        if args.fmt == "faiss":
            with open(in_dir / f"{x}.{args.suffix}", "r") as f_in,\
                open(out_dir / f"{x}.lengths", "w") as f_out:
                sents[x] = []
                for line in f_in:
                    sent = line.rstrip("\n").split() 
                    f_out.write(f"{len(sent)}\n")
                    for w in sent:
                        if not w in vocab:
                            vocab.append(w)
                    sents[x].append(sent)
        elif args.fmt == "sklearn":
            copyfile(in_dir / f"../{x}.lengths", out_dir / f"{x}.lengths")
            with open(in_dir / f"{x}.{args.suffix}", "r") as f_in,\
                open(in_dir / f"../{x}.lengths", "r") as f_len:
                lines = f_in.read().strip().split("\n")
                sizes = f_len.read().strip().split("\n")
                sizes = list(map(int, sizes))
                assert sum(sizes) == len(lines)
                sents[x] = []
                offset = 0
                for size in sizes:
                    sent = lines[offset:offset+size]
                    for w in sent:
                        if not w in vocab:
                            vocab.append(w)
                    sents[x].append(sent)
                    offset += size
        else:
            raise ValueError(f"Unknown format: {len(args.fmt)}")
    
    n = len(vocab)
    print(f"Vocab size: {n}")
    for x in ["train", "valid"]:
        feats = []
        if args.save_int:
            feats = np.asarray(
                [
                    vocab.index(w) for sent in sents[x] for w in sent
                ]
            )[:, np.newaxis]
        else:
            for sent in sents[x]:
                feat = np.stack(
                    [
                        np.eye(n)[vocab.index(w)] 
                        for w in sent
                    ]
                )
                feats.append(feat)
            feats = np.concatenate(feats)
        np.save(out_dir / f"{x}.npy", feats)

if __name__ == "__main__":
    main()
