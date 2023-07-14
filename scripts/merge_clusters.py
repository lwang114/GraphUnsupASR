#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np
import tqdm
import torch
import random
from shutil import copyfile

from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    # fmt: off
    parser.add_argument('source', help='directory with features')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--cluster-dir', help='where the clusters are')
    parser.add_argument('--pooling', type=str, default='mean', choices=["mean", "sample", "sum", "concat"], help='how to pool')
    parser.add_argument('--fmt', default='faiss', choices={"sklearn", "faiss"})
    parser.add_argument('--max_segment_length', default='max_segment_length.txt', type=str)
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split)
    cluster_path = osp.join(args.cluster_dir, args.split + ".src")
    print(f"data path: {source_path}")

    features = np.load(source_path + ".npy", mmap_mode="r")
    sizes = []
    offsets = []
    offset = 0
    with open(source_path + ".lengths", "r") as len_f:
        for line in len_f:
            length = int(line.rstrip())
            sizes.append(length)
            offsets.append(offset)
            offset += length

    clusters = []
    if args.fmt == "sklearn":
        with open(cluster_path, "r") as cf:
            lines = cf.read().strip().split("\n")

            for size, offset in zip(sizes, offsets):
                items = lines[offset:offset+size]
                items = list(map(int, items))
                clusters.append(items)
    else:
        with open(cluster_path, "r") as cf:
            for line, size in zip(cf, sizes):
                line = line.rstrip()
                items = line.split()
                if len(items) < size:
                    items.append(items[-1])
                else:
                    items = items[:size]
                items = list(map(int, items))
                clusters.append(items)
    if osp.exists(args.max_segment_length):
        with open(args.max_segment_length, "r") as f:
            l_max = int(f.read().strip())
    elif args.pooling == "concat":
        l_max = int(args.max_segment_length)
    else:
        l_max = -1
    print(f"Max segment length: {l_max}")  # XXX
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, args.split)

    copyfile(source_path + ".tsv", save_path + ".tsv")

    if os.path.exists(source_path + ".phn"):
        copyfile(source_path + ".phn", save_path + ".phn")
    if os.path.exists(osp.join(args.source, "dict.phn.txt")):
        copyfile(
            osp.join(args.source, "dict.phn.txt"),
            osp.join(args.save_dir, "dict.phn.txt"),
        )
    if os.path.exists(source_path + ".wrd"):
        copyfile(source_path + ".wrd", save_path + ".wrd")

    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy")
    
    def merge(feats, clust):
        feats = torch.from_numpy(feats.copy())
        d = feats.size(-1)
        clust = torch.LongTensor(clust)
        _, counts = clust.unique_consecutive(return_counts=True)
        curr = 0

        merged = []
        for c in counts:
            c = c.item()
            start = curr
            end = curr + c
            curr += c
            if args.pooling == "mean":
                new_x = feats[start:end].mean(dim=0)
            elif args.pooling == "sum":
                new_x = feats[start:end].sum(dim=0)
            elif args.pooling == "concat":
                new_x = feats.new_zeros(l_max*d)
                new_x[:c*d] = feats[start:end].flatten() / c
            elif args.pooling == "sample":
                new_x = feats[start + int(random.random() * c)]
            else:
                raise NotImplementedError()
            merged.append(new_x)
        return torch.stack(merged, dim=0).numpy()

    with open(save_path + ".lengths", "w") as l_f:
        for size, offset, clust in tqdm.tqdm(
            zip(sizes, offsets, clusters), total=len(sizes)
        ):
            end = size + offset
            feats = features[offset:end]
            feats = merge(feats, clust)
            print(len(feats), file=l_f)
            npaa.append(feats)


if __name__ == "__main__":
    main()
