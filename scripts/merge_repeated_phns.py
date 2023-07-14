import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.in_path, "r") as f_in,\
        open(args.out_path, "w") as f_out:
        for line in f_in:
            phns = line.strip().split()
            new_phns = []
            for phn, next_phn in zip(phns[:-1], phns[1:]):
                if phn != next_phn:
                    new_phns.append(phn)
            new_phns.append(next_phn)
            f_out.write(" ".join(new_phns)+"\n")
        

if __name__ == "__main__":
    main()
