import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("ref_path")
    parser.add_argument("out_path")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.in_path, "r") as fin,\
        open(args.ref_path, "r") as fref,\
        open(args.out_path, "w") as fout:
        for inline, refline in zip(fin, fref):
            in_tokens = inline.strip().split()
            token_idx = 0
            ref_tokens = refline.strip().split()
            out_tokens = []
            for t in ref_tokens:
                if t == "<SIL>":
                    out_tokens.append("<SIL>")
                else:
                    out_tokens.append(in_tokens[token_idx])
                    token_idx += 1
            fout.write(" ".join(out_tokens)+"\n")

if __name__ == "__main__":
    main()
