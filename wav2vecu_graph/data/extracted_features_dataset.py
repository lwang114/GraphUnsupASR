# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import contextlib

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


class ExtractedFeaturesDataset(FairseqDataset):
    def __init__(
        self,
        path,
        split,
        min_length=3,
        max_length=None,
        labels=None,
        label_dict=None,
        shuffle=True,
        sort_by_length=True,
        aux_target_postfix=None,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.label_dict = label_dict

        if labels is not None:
            assert label_dict is not None

        self.sizes = []
        self.offsets = []
        self.segments = []
        self.gt_segments = []
        self.skip_indices = []
        self.labels = []
        self.aux_tgt = None

        path = os.path.join(path, split)
        data_path = path
        self.data = np.load(data_path + ".npy", mmap_mode="r")
        self.clus_data = None
        if os.path.exists(data_path + "_clus.npy"):
            self.clus_data = np.load(data_path + "_clus.npy", mmap_mode="r")

        offset = 0
        skipped = 0

        if not os.path.exists(path + f".{labels}"):
            labels = None

        with open(data_path + ".lengths", "r") as len_f, open(
            path + f".{labels}", "r"
        ) if labels is not None else contextlib.ExitStack() as lbl_f:
            for line in len_f:
                length = int(line.rstrip())
                lbl = None if labels is None else next(lbl_f).rstrip().split()
                if length >= min_length and (
                    max_length is None or length <= max_length
                ):
                    self.sizes.append(length)
                    self.offsets.append(offset)
                    if lbl is not None:
                        self.labels.append(lbl)
                offset += length

        if os.path.exists(data_path + ".src"):
            with open(data_path + ".src", "r") as src_f:
                for idx, line in enumerate(src_f):
                    clusts = list(map(int, line.rstrip().split()))
                    clusts = torch.tensor(clusts)
                    units, _, counts = clusts.unique_consecutive(return_inverse=True, return_counts=True)
                    segments = []
                    skip_indices = []
                    offset = 0
                    for u, c in zip(units[:-1], counts[:-1]):
                        if offset >= self.sizes[idx]:
                            continue
                        if u >= 0:                
                            offset += c.item()                
                            segments.append(offset)
                        else:
                            skip_indices.extend(list(range(offset, offset+c.item())))
                            offset += c.item()
                    self.segments.append(segments)
                    self.skip_indices.append(skip_indices)

        if os.path.exists(data_path + "_gt.src"):
            with open(data_path + "_gt.src", "r") as src_f:
                for idx, line in enumerate(src_f):
                    clusts = list(map(int, line.rstrip().split()))
                    clusts = torch.tensor(clusts)
                    _, _, counts = clusts.unique_consecutive(return_inverse=True, return_counts=True)
                    segments = []
                    offset = counts[0].item()
                    for c in counts[1:]:
                        if offset >= self.sizes[idx]:
                            continue
                        segments.append(offset)
                        offset += c.item()
                    self.gt_segments.append(segments)

        self.sizes = np.asarray(self.sizes)
        self.offsets = np.asarray(self.offsets)
        
        if aux_target_postfix is not None:
            if not os.path.exists(path+f".{aux_target_postfix}"):
                logger.info(f"auxaliry target for {split} missing")
            else:
                with open(path+f".{aux_target_postfix}", "r") as t_f:
                    self.aux_tgt = [
                        torch.LongTensor(list(map(int,seg.strip().split())))\
                                    for seg in t_f]
 
        logger.info(f"loaded {len(self.offsets)}, skipped {skipped} samples")

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.data[offset:end].copy()).float()
        clus_feats = None
        if self.clus_data is not None:
            clus_feats = torch.from_numpy(self.clus_data[offset:end].copy()).float()
        bin_labels = None
        if len(self.segments):
            bin_labels = torch.zeros(self.sizes[index])
            bin_labels[self.segments[index]] = 1.0
            if len(self.skip_indices[index]):
                bin_labels[self.skip_indices[index]] = 0.5

        gt_bin_labels = None
        if len(self.gt_segments):
            gt_bin_labels = torch.zeros(self.sizes[index])
            gt_bin_labels[self.gt_segments[index]] = 1.0


        res = {"id": index, "features": feats, "clus_features": clus_feats, "bin_labels": bin_labels, "gt_bin_labels": gt_bin_labels}
        if len(self.labels) > 0:
            res["target"] = self.label_dict.encode_line(
                self.labels[index],
                line_tokenizer=lambda x: x,
                append_eos=False,
            )
        
        if self.aux_tgt:
            res["aux_target"] = self.aux_tgt[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        clus_features = [s["clus_features"] for s in samples]
        sizes = [len(s) for s in features]
        bin_labels = [s["bin_labels"] for s in samples]
        gt_bin_labels = [s["gt_bin_labels"] for s in samples]

        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        collated_clus_features = None
        if clus_features[0] is not None:
            collated_clus_features = clus_features[0].new_zeros(
                len(clus_features), target_size, clus_features[0].size(-1)
            )

        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        
        collated_bin_labels = None
        if bin_labels[0] is not None:
            collated_bin_labels = bin_labels[0].new_zeros(
                len(bin_labels), target_size
            )

        collated_gt_bin_labels = None
        if gt_bin_labels[0] is not None:
            collated_gt_bin_labels = gt_bin_labels[0].new_zeros(
                len(gt_bin_labels), target_size
            )

        for i, (f, cf, size, b, gb) in enumerate(zip(features, clus_features, sizes, bin_labels, gt_bin_labels)):
            collated_features[i, :size] = f
            if cf is not None:
                collated_clus_features[i, :size] = cf
            padding_mask[i, size:] = True
            if b is not None:
                collated_bin_labels[i, :size] = b
            if gb is not None:
                collated_gt_bin_labels[i, :size] = gb

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                "features": collated_features,
                "clus_features": collated_clus_features,
                "padding_mask": padding_mask,
                "bin_labels": collated_bin_labels,
                "gt_bin_labels": collated_gt_bin_labels,
            }
        }

        if len(self.labels) > 0:
            target = data_utils.collate_tokens(
                [s["target"] for s in samples],
                pad_idx=self.label_dict.pad(),
                left_pad=False,
            )
            res["target"] = target
        
        if self.aux_tgt:
            idxs = torch.nn.utils.rnn.pad_sequence(
                [s["aux_target"] for s in samples],
                batch_first=True,
                padding_value=-1,
            )
            res["net_input"]["aux_target"] = idxs
        
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]
