#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np

from sklearn.decomposition import PCA



def get_parser():
    parser = argparse.ArgumentParser(
        description="compute a pca matrix given an array of numpy features"
    )
    # fmt: off
    parser.add_argument('data', help='numpy file containing features')
    parser.add_argument('--output', help='where to save the pca matrix', required=True)
    parser.add_argument('--dim', type=int, help='dim for pca reduction', required=True)
    parser.add_argument('--eigen-power', type=float, default=0, help='eigen power, -0.5 for whitening')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print("Reading features")
    x = np.load(args.data, mmap_mode="r")

    print("Computing PCA")
    pca = PCA(
        n_components=args.dim,
    )
    pca.fit(x)
    
    
    b = pca.mean_
    A = pca.components_
    print(f"A.shape: {A.shape}, explained variance ratio: {pca.explained_variance_ratio_}")

    os.makedirs(args.output, exist_ok=True)

    prefix = str(args.dim)

    np.save(osp.join(args.output, f"{prefix}_pca_A"), A.T)
    np.save(osp.join(args.output, f"{prefix}_pca_b"), b)


if __name__ == "__main__":
    main()
