import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("wrd_path")
    parser.add_argument("phn_path")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.in_path, "r") as fin,\
        open(args.wrd_path, "w") as fwrd,\
        open(args.phn_path, "w") as fphn:
        for line in fin:
            words = line.rstrip().split()
            wrd_ints = []
            phn_ints = []
            for w in words:
                wrd_int = [
                    str(ord(c.upper())-ord("A"))
                    for c in w if c.isalpha()
                ]
                wrd_ints.append(",".join(wrd_int))
                phn_ints.extend(wrd_int)
            fwrd.write(" ".join(wrd_ints)+"\n")
            fphn.write(" ".join(phn_ints)+"\n")

if __name__ == "__main__":
    main()
