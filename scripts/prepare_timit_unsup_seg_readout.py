import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir")
    parser.add_argument("--tsv-path")
    parser.add_argument("--out-path")
    parser.add_argument("--downsample-rate", type=int, default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)
    x = Path(args.tsv_path).stem

 
    with open(in_dir / f"{x}.src", "r") as f_in,\
        open(out_path, "w") as f_out:
            for line in f_in:
                fpath = Path(line.split()[0])
                phns = " ".join(line.split()[1:])
                if args.downsample_rate > 1:
                    phns = phns[::args.downsample_rate]
                print(phns, file=f_out)

if __name__ == "__main__":
    main()
