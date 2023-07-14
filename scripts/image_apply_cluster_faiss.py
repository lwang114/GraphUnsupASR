import argparse
import numpy as np
from pathlib import Path
import sys
import tqdm
import torch

import faiss
from image_cluster_faiss import parse_faiss_specs


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
    spec = path.stem
    
    try:
        faiss_spec = parse_faiss_specs(spec)[0]
    except:
        print(spec)
        raise

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca:
        A = torch.from_numpy(np.load(path / "pca_A.npy"))
        b = torch.from_numpy(np.load(path / "pca_b.npy"))
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(path / "centroids.npy")
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    #res = faiss.StandardGpuResources()
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1])
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    )
    faiss_index = index_flat # faiss.index_cpu_to_gpu(res, 0, index_flat)
    faiss_index.add(centroids)

    feats = np.load(data / f"{args.split}.npy")
    feats = torch.from_numpy(feats)
    with torch.no_grad():
        with open(path / f"{args.split}.src", "w") as fp:
            if faiss_spec.pca:
                feats = torch.mm(feats, A) + b
            if faiss_spec.norm:
                feats = F.normalize(feats, p=2, dim=-1)
            
            feats = feats.cpu().numpy()

            _, z = faiss_index.search(feats, 1)
            if args.fmt == "sklearn":
                print("\n".join(str(x.item()) for x in z), file=fp)
            else:
                offset = 0
                with open(data / f"{args.split}.lengths", "r") as fl:
                    z = [str(x.item()) for x in z]
                    lengths = fl.read().strip().split("\n")
                    lengths = list(map(int, lengths))
                    for size in lengths:
                        print(
                            " ".join(z[offset:offset+size]), 
                            file=fp,
                        )
                        offset += size        

if __name__ == "__main__":
    main()
