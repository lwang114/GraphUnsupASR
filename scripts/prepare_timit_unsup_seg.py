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
    tsv_path = Path(args.tsv_path)

    with open(tsv_path, "r") as f_tsv:
        lines = f_tsv.read().strip().split("\n")
        _ = lines.pop(0)
        uids = [
            l.split()[0].split("\t")[0].split("/")[-1].split(".")[0] 
            for l in lines
        ]
    
    info = dict()
    info_list = []
    for x in ["train", "test"]:
        with open(in_dir / f"{x}.src", "r") as f_in:
            for line in f_in:
                fpath = Path(line.split()[0])
                phns = " ".join(line.split()[1:])
                fname = fpath.name.split(".wav")[0]
                spk = fpath.parent.name
                uid = f"{spk}_{fname}"
                info[uid] = phns
                info_list.append(phns)

    with open(out_path, "w") as f_out: 
        for idx, uid in enumerate(uids):
            phns = info[uid]
            if args.downsample_rate > 1:
                phns = phns[::args.downsample_rate]
            print(phns, file=f_out)


if __name__ == "__main__":
    main()
