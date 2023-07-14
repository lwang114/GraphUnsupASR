import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import sys

def get_parser():
    parser = argparse.ArgumentParser(description="apply clusters")
    
    parser.add_argument("data", help="location of npy files")
    parser.add_argument(
        "--split",
        help="split to process", 
        required=True,
    )
    parser.add_argument(
        "--path",
        help="path to pca and centroids", 
        required=True,
    )
    parser.add_argument(
        "--fmt", 
        choices={"sklearn", "faiss"}, 
        default="faiss",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    data = Path(args.data)
    path = Path(args.path)

    centroids = np.load(path / "centroids.npy")
    print("Loaded centroids", centroids.shape, file=sys.stderr, flush=True)
 
    kmeans = KMeans(
        n_clusters=centroids.shape[0],
    )
    kmeans.cluster_centers_ = centroids
    kmeans._n_threads = 1

    feats = np.load(data / f"{args.split}.npy")
    with open(path / f"{args.split}.src", "w") as fp:        
        z = kmeans.predict(feats)
        if args.fmt == "sklearn":
            print("\n".join(str(x) for x in z), file=fp)
        else:
            offset = 0
            with open(data / f"{args.split}.lengths", "r") as fl:
                z = list(map(str, z))
                lengths = fl.read().strip().split("\n")
                lengths = list(map(int, lengths))
                for size in lengths:
                    print(
                        " ".join(z[offset:offset+size])+"\n", 
                        file=fp,
                    )
                    offset += size
            
if __name__ == "__main__":
    main()
