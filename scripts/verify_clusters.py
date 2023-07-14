import argparse
import numpy as np


# Verify if a sequence of cluster units is extracted from a sequence of continuous features
def load_clus(src_file):
    clus = []
    with open(src_file, "r") as f:
        for l in f:
            clus.extend(
                map(int, l.strip().split())
            )
    return np.asarray(clus)

def load_feat(npy_file, len_file):
    feat = np.load(npy_file)
    offsets = []
    offset = 0
    with open(len_file, "r") as f_len:
        sizes = f_len.read().strip().split("\n")
        sizes = list(map(int, sizes))
        for size in sizes:
            offsets.append(offset)
            offset += size
    return feat, offsets, sizes 

def assign(feat, centroids, offsets, sizes):
    dist = ((feat[:, np.newaxis] - centroids[np.newaxis])**2).sum(-1)
    preds = dist.argmin(-1)
    return preds, dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", default="manifest/timit_norep/matched/feat/CLUS128/valid.src")
    parser.add_argument("--npy_file", default="manifest/timit_norep/matched/feat/valid.npy")
    parser.add_argument("--length_file", default="manifest/timit_norep/matched/feat/valid.lengths")
    parser.add_argument("--centroid_file", default="manifest/timit_norep/matched/feat/CLUS128/centroids.npy")
    args = parser.parse_args()

    clus = load_clus(args.src_file)
    feat, offsets, sizes = load_feat(args.npy_file, args.length_file)
    centroids = np.load(args.centroid_file)

    preds, dist = assign(feat, centroids, offsets, sizes)
    acc = (preds == clus).mean()

    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    main()
