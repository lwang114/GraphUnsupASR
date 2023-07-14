import argparse
import fairseq
import faiss
import gc
import random
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
import tqdm
import torch

from collections import namedtuple



def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from image feats"
    )
    parser.add_argument("data", help="location of the tsv file")
    parser.add_argument(
        "--save-dir", 
        help="where to save the output", 
        required=True,
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--sample-pct", 
        "-r", 
        type=float,
        help='percentage of timesteps to sample', 
        default=0,
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    data = Path(args.data)

    feats = np.load(data)
    save_path = save_dir / f"CLUS{args.n_clusters}"
    save_path.mkdir(parents=True, exist_ok=True)
    d = feats.shape[-1]
    x = feats

    print("Computing kmeans")
    kmeans = KMeans(
        args.n_clusters,
        init="k-means++",
        verbose=1,
    ).fit(x)
    z = kmeans.predict(x)
    np.save(save_path / "centroids.npy", kmeans.cluster_centers_)
    # Save labels
    with open(save_path / "train.src", "w") as fp:
        print("\n".join(str(z_) for z_ in z), file=fp)


if __name__ == "__main__":
    main()
             
