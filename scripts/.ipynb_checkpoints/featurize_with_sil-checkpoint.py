import argparse
import json
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(
        description="Adding optional silences around in between word boundaries of feature vectors"
    )
    parser.add_argument("in_prefix")
    parser.add_argument("out_prefix")
    parser.add_argument(
        "--sil-prob",
        "-s",
        type=float,
        default=0,
        help="probability of inserting silence between each word",
    )
    parser.add_argument(
        "--alignment",
        help=".jsonlines files containing word and phoneme alignments",
        required=True,
    )
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    in_prefix = args.in_prefix
    out_prefix = args.out_prefix
    sil_prob = args.sil_prob
    sil = "<SIL>"

    in_feats = np.load(in_prefix+".npy")
    bsz = in_feats.shape[0]
    in_feats = np.concatenate(
        (np.zeros((bsz, 1)), in_feats), axis=1,
    )
    d = in_feats.shape[-1]
    
    with open(in_prefix+".lengths", "r") as fi:
        lengths = fi.read().strip().split("\n")
        lengths = list(map(int, lengths))
    
    out_feats = []
    with open(args.alignment, "r") as fa:
        offset = 0
        out_feats = []
        out_lengths = []
        sil = np.eye(d)[np.newaxis, 0]
        for line, in_length in zip(fa, lengths):
            ali = json.loads(line.rstrip("\n"))
            starts = ali["phone_starts"]
            ends = ali["phone_ends"]
            sample_sil_probs = None
            if sil_prob > 0 and len(starts) > 1:
                sample_sil_probs = np.random.random(len(starts) - 1)
            
            out_length = len([s for start in starts for s in start])
            if not in_length == out_length:
                print(f"Warning: feature {in_length} != alignment {out_length}")
            for i, start in enumerate(starts):
                out_feats.append(in_feats[offset:offset+len(start)])
                if (
                    sample_sil_probs is not None
                    and i < len(sample_sil_probs)
                    and sample_sil_probs[i] < sil_prob
                ):
                    out_feats.append(sil)
                    out_length += 1
                offset += len(start) 
            out_lengths.append(out_length)
    out_feats = np.concatenate(out_feats)
    np.save(out_prefix+".npy", out_feats)
    with open(out_prefix+".lengths", "w") as fo:
        out_lengths = list(map(str, out_lengths))
        fo.write("\n".join(out_lengths))


if __name__ == "__main__":
    main()
