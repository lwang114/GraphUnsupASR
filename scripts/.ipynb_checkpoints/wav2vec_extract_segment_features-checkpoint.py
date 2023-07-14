#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import os.path as osp
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from shutil import copyfile
import numpy as np
from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score
np.random.seed(1)
torch.manual_seed(1)

def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)
    parser.add_argument('--layer', type=int, default=14, help='which layer to use')
    parser.add_argument('--classify', action='store_true')
    # fmt: on
    return parser


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc, ali):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source, mask=False, features_only=True, layer=self.layer)
            rawfeats =  m_res["x"].squeeze(0).cpu()
            feats = []
            for begin, end in zip(ali["begins"], ali["ends"]):
                b = int(begin*50)
                e = max(int(end*50), b+1)
                if b > len(rawfeats):
                    print(f"Warning: {loc} begin time frame {b} > total number of frames {len(rawfeats)}")
                    b = len(rawfeats) - 1
                feats.append(rawfeats[b:e].mean(0))
            return torch.stack(feats)


def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp,\
        open(osp.join(args.data, args.split) + ".jsonlines", "r") as fpa:
        lines = fp.read().split("\n")
        ali_lines = fpa.read().strip().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        alis = [json.loads(line) for line in ali_lines]
        assert len(files) == len(alis)

        num = len(files)
        reader = Wav2VecFeatureReader(args.checkpoint, args.layer)

        def iterate():
            for fname, ali in zip(files, alis):
                w2v_feats = reader.get_feats(fname, ali)
                yield w2v_feats

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()
    """
    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv")
        if osp.exists(osp.join(args.data, args.split) + ".wrd"):
            copyfile(osp.join(args.data, args.split) + ".wrd", dest + ".wrd")
        if osp.exists(osp.join(args.data, args.split) + ".phn"):
            copyfile(osp.join(args.data, args.split) + ".phn", dest + ".phn")

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = osp.join(args.save_dir, args.split)
    # Load segment features
    npaa = create_files(save_path)

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for w2v_feats in tqdm.tqdm(iterator, total=num):
            print(len(w2v_feats), file=l_f)

            if len(w2v_feats) > 0:
                npaa.append(w2v_feats.numpy())
    """
    if args.classify:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_train = torch.FloatTensor(
            np.load(osp.join(args.save_dir, "train.npy"))
        ).to(device)
        X_test = torch.FloatTensor(
            np.load(osp.join(args.save_dir, "test.npy"))
        ).to(device)
        train_labels = []
        test_labels = []
        vocab = []
        with open(osp.join(args.save_dir, "train.wrd"), "r") as f_tr,\
            open(osp.join(args.save_dir, "test.wrd"), "r") as f_tx:
            for sent in f_tr:
                for w in sent.rstrip("\n").split():
                    if not w in vocab:
                        train_labels.append(len(vocab))
                        vocab.append(w)
                    else:
                        train_labels.append(vocab.index(w))
            
            for sent in f_tx:
                for w in sent.rstrip("\n").split():
                    if not w in vocab:
                        print(f"Warning: {w} not in training set", flush=True)
                        test_labels.append(len(vocab))
                        vocab.append(w)
                    else:
                        test_labels.append(vocab.index(w))
        y_train = torch.LongTensor(train_labels).to(device)
        y_test = torch.LongTensor(test_labels).to(device)
        assert (X_train.size(0) == y_train.size(0)) and (X_test.size(0) == y_test.size(0))
        clf = nn.Linear(1024, len(vocab))
        clf = clf.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            clf.parameters(),
            lr=0.001,
        )
        for epoch in range(100):
            logits = clf(X_train)
            loss = criterion(logits, y_train)
            print(f"Epoch {epoch}, training loss: {loss:.3f}", flush=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            y_pred = clf(X_test).argmax(-1).detach().cpu().numpy()
            acc = accuracy_score(y_test.cpu(), y_pred)
            micro_f1 = f1_score(y_test.cpu(), y_pred, average="micro")
            macro_f1 = f1_score(y_test.cpu(), y_pred, average="macro")
            print(
                f"Validation Epoch {epoch}, accuracy: {acc:.4f}\tmicro F1: {micro_f1:.4f}\tmacro F1: {macro_f1:.4f}", 
                flush=True,
            )
        
if __name__ == "__main__":
    main()
