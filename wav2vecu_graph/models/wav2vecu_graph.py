# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto
import math
import numpy as np
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    SamePad,
    TransposeLast,
)
import pdb
from .utils import *


class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()
    CPC = auto()
    BINARY = auto()

@dataclass
class PeakDetectionConfig(FairseqDataclass):
    prominence: float = 0.05
    
@dataclass
class SegmentationConfig(FairseqDataclass):
    type: SegmentationType = SegmentationType.NONE
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    soft_pool_join: bool = False
    remove_zeros: bool = False
    in_dim: int = 0  # Trainable segmenter only
    latent_dim: int = 0  # Trainable segmenter only
    n_negatives: int = 1  # Trainable segmenter only
    pos_weight: float = 1  # Binary segmenter only
    ignore_value: float = 0.5 # Binary segmenter only
    smoothness_weight: float = 0.0 # Binary segmenter only
    batch_shuffle: bool = False  # CPC segmenter only
    peak_detection: PeakDetectionConfig = PeakDetectionConfig()  # CPC segmenter only


@dataclass
class Wav2vecU_GraphConfig(FairseqDataclass):
    gan_type: str = "L1"
    discriminator_type: str = "cnn"
    generator_type: str = "cnn"
    generator_input_type: str = "float"
    reset_discriminator_every_update: bool = False
    skipgram_size: int = 1
    skipgram_only: bool = False
    position_skipgram: bool = False
    trigram_size: int = 0
    no_silence: bool = False
    no_special_tokens: bool = False
    
    discriminator_kernel: int = 3
    discriminator_dilation: int = 1
    discriminator_dim: int = 256
    discriminator_causal: bool = True
    discriminator_linear_emb: bool = False
    discriminator_depth: int = 1
    discriminator_max_pool: bool = False
    discriminator_act_after_linear: bool = False
    discriminator_dropout: float = 0.0
    discriminator_spectral_norm: bool = False
    discriminator_weight_norm: bool = False

    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_pad: int = -1
    generator_bias: bool = False
    generator_dropout: float = 0.0
    generator_batch_norm: int = 0
    generator_residual: bool = False
    generator_classifier: bool = False
    generator_avg_pool_kernel: int = 0
    generator_avg_pool_stride: int = 1
    
    quantizer_type: str = "gumbel"
    quantizer_input_dim: int = 1024
    quantizer_codebook_size: int = 0
    quantizer_decode_objective: str = "cross_entropy"
    quantizer_weight: float = 1.0

    blank_weight: float = 0
    blank_mode: str = "add"
    blank_is_sil: bool = False
    no_softmax: bool = False

    smoothness_weight: float = 0.0
    smoothing: float = 0.0
    smoothing_one_sided: bool = False
    gradient_penalty: float = 0.0
    probabilistic_grad_penalty_slicing: bool = False
    code_penalty: float = 0.0
    mmi_weight: float = 0.0
    segment_weight: float = 0.0
    target_dim: int = 64
    target_downsample_rate: int = 2
    gumbel: bool = False
    hard_gumbel: bool = True
    temp: Tuple[float, float, float] = (2, 0.1, 0.99995)
    input_dim: int = 128
    hidden_dim: int = 256

    segmentation: SegmentationConfig = SegmentationConfig()


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class RandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        target_num = math.ceil(dense_x.size(1) * self.subsample_rate)
        ones = torch.ones(dense_x.shape[:-1], device=dense_x.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, dense_x.size(-1))
        dense_x = dense_x.gather(1, indices_ld)
        dense_padding_mask = dense_padding_mask.gather(1, index=indices)
        return dense_x, dense_padding_mask


class UniformRandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        bsz, tsz, fsz = dense_x.shape

        target_num = math.ceil(tsz * self.subsample_rate)

        rem = tsz % target_num

        if rem > 0:
            dense_x = F.pad(dense_x, [0, 0, 0, target_num - rem])
            dense_padding_mask = F.pad(
                dense_padding_mask, [0, target_num - rem], value=True
            )

        dense_x = dense_x.view(bsz, target_num, -1, fsz)
        dense_padding_mask = dense_padding_mask.view(bsz, target_num, -1)

        if self.cfg.mean_pool:
            dense_x = dense_x.mean(dim=-2)
            dense_padding_mask = dense_padding_mask.all(dim=-1)
        else:
            ones = torch.ones((bsz, dense_x.size(2)), device=dense_x.device)
            indices = ones.multinomial(1)
            indices = indices.unsqueeze(-1).expand(-1, target_num, -1)
            indices_ld = indices.unsqueeze(-1).expand(-1, -1, -1, fsz)
            dense_x = dense_x.gather(2, indices_ld).reshape(bsz, -1, fsz)
            dense_padding_mask = dense_padding_mask.gather(2, index=indices).reshape(
                bsz, -1
            )
        return dense_x, dense_padding_mask


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class CPCSegmenter(Segmenter):
    # Code adapted from https://github.com/felixkreuk/UnsupSeg
    def __init__(self, cfg: SegmentationConfig):
        super().__init__(cfg)
        I_DIM = cfg.in_dim
        LS = cfg.latent_dim
        self.enc = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                I_DIM,  
                LS,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            TransposeLast(),
            nn.Linear(LS, LS), 
        )
    
    def score(self, f, b):
        return F.cosine_similarity(f, b, dim=-1)
    
    def forward(self, logits, padding_mask):
        device = logits.device
        z = self.enc(logits)
        pos_pred = self.score(z[:, :-1], z[:, 1:])
        preds = [pos_pred] 
        for _ in range(self.cfg.n_negatives):
            time_reorder = torch.randperm(pos_pred.shape[1])
            batch_reorder = torch.arange(pos_pred.shape[0])
            if self.cfg.batch_shuffle:
                batch_reorder = torch.randperm(pos_pred.shape[0])
            neg_pred = self.score(z[:, :-1], z[batch_reorder][:, time_reorder])
            preds.append(neg_pred)
        
        out = torch.stack(preds, dim=-1)
        out = F.log_softmax(out, dim=-1)
        loss = - out[...,0] * (1. - padding_mask[...,:-1].float())
        return out, loss
    
    def logit_segment(self, logits, padding_mask):
        pred_scores, _ = self(logits, padding_mask)
        pred_scores = pred_scores[...,0]
        pred_scores = replicate_first_k_frames(pred_scores, k=1, dim=1)
        pred_scores = 1 - max_min_norm(pred_scores)
        lengths = (1 - padding_mask.long()).sum(-1)
        
        peaks = detect_peaks(
            x=pred_scores,
            lengths=lengths,
            prominence=self.cfg.peak_detection.prominence,
            width=None,
            distance=None,
        )
        
        preds = logits.new_zeros(logits.size(0), logits.size(1))
        for i, peak in enumerate(peaks):
            preds[i, peak] = 1.
        preds = preds.cumsum(-1)
            
        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)
            
            new_logits[b].index_add_(
                dim=0, index=idx.to(new_logits.device), source=logits[b]
            )
            new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class BinarySegmenter(Segmenter):
    def __init__(self, cfg: SegmentationConfig):
        super().__init__(cfg)
        self.classifier = nn.Sequential(
            # nn.LogSoftmax(dim=-1),
            TransposeLast(),
            nn.Conv1d(cfg.in_dim, 512, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0),
            TransposeLast(),
        )
        self.pos_weight = cfg.pos_weight
        self.ignore_value = cfg.ignore_value
        self.smoothness_weight = cfg.smoothness_weight
    
    def forward(self, inputs, padding_mask, labels=None):
        logits, feats = inputs
        preds = self.classifier(feats)
        preds = preds[...,0]
        preds[padding_mask] = 0.0
        if labels is not None:
            loss = self.binary_cross_entropy_with_logits(
                preds, labels, 
                padding_mask, 
                pos_weight=self.pos_weight,
                ignore_value=self.ignore_value,
            )
            # Smoothness loss
            if self.smoothness_weight > 0:
                logit_preds = -(logits[:, :-1] * logits[:, 1:]).sum(-1)
                smoothness_loss = self.binary_cross_entropy_with_logits(
                    logit_preds, labels[:, :-1],
                    padding_mask[:, :-1],
                    pos_weight=self.pos_weight,
                    ignore_value=self.ignore_value,
                )
                loss += self.smoothness_weight * smoothness_loss
            return preds, loss
        return preds
    
    def binary_cross_entropy_with_logits(
            self, input, 
            target, 
            padding_mask,
            pos_weight=1.0, 
            ignore_value=0.5,
        ):
        pos_weight = pos_weight * input.new_ones(1)
        loss = F.binary_cross_entropy_with_logits(
            input, target, reduction="none", pos_weight=pos_weight,
        )
        loss[padding_mask] = 0.
        if self.ignore_value is not None: 
            loss[target == ignore_value] = 0.
        loss = loss.sum()
        return loss

    def logit_segment(self, inputs, padding_mask):
        logits, feats = inputs
        pred_scores = self(inputs, padding_mask)
        preds = (pred_scores > 0).long().cumsum(-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(
                    return_inverse=True, return_counts=True
                )
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join: 
                if self.cfg.soft_pool_join:
                    # compute soft counts 
                    sc = pred_scores[b].sigmoid().cumsum(-1)
                    
                    # construct pooling matrix of size (N_seg, T)
                    ns = u.numel()
                    w = torch.arange(ns).repeat(tsz, 1).t()
                    w = w.to(pred_scores.device)
                    w = - torch.abs(w - sc) * 10.
                    w[padding_mask[b].repeat(ns, 1)] = -1e14
                    w = torch.softmax(w, dim=-1)
                    
                    # perform pooling on logits
                    new_logits[b, :ns] = torch.mm(w, logits[b]) 
                else:
                    u[0] = 0
                    u[1:] = c.cumsum(0)[:-1]
                    m = c > 1
                    r = torch.rand(m.sum())
                    o = (c[m] * r).long()
                    u[m] += o
                    new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class UniformRandomJoinSegmenter(UniformRandomSegmenter, JoinSegmenter):
    pass


SEGMENT_FACTORY = {
    SegmentationType.NONE: Segmenter,
    SegmentationType.RANDOM: RandomSegmenter,
    SegmentationType.UNIFORM_RANDOM: UniformRandomSegmenter,
    SegmentationType.UNIFORM_RANDOM_JOIN: UniformRandomJoinSegmenter,
    SegmentationType.JOIN: JoinSegmenter,
    SegmentationType.BINARY: BinarySegmenter,
}


class Discriminator(nn.Module):
    def __init__(self, dim, cfg: Wav2vecU_GraphConfig):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        dilation = cfg.discriminator_dilation
        self.max_pool = cfg.discriminator_max_pool

        if cfg.discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if cfg.discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif cfg.discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
                nn.Dropout(cfg.discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(cfg.discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
        ]

        if cfg.discriminator_linear_emb:
            emb_net = [make_conv(dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
            ]

        if cfg.discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(cfg.discriminator_dropout),
            *inner_net,
        )

    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz
        return x


class MLPDiscriminator(nn.Module):
    def __init__(self, dim, cfg: Wav2vecU_GraphConfig):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        depth = cfg.discriminator_depth
        self.depth = depth

        def make_mlp(in_d, hid_d, out_d, depth):
            if depth <= 0:
                return nn.Linear(in_d, out_d)
            layers = [
                nn.Linear(in_d, hid_d),
                nn.GELU(),
                nn.Dropout(cfg.discriminator_dropout),
            ]
            for _ in range(depth - 1):
                layers.append(nn.Linear(hid_d, hid_d))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hid_d, out_d))
            layers = nn.Sequential(*layers)
            return layers

        self.dropout = nn.Dropout(cfg.discriminator_dropout)
        self.layers = nn.ModuleList(
            [make_mlp(kernel, 128, 1, depth) for _ in range(inner_dim)]
        )

    def forward(self, x, padding_mask, reduction="sum"):
        #x = self.net(x).squeeze(-1)
        assert x.size(1) <= len(self.layers)
        x = torch.stack(
            [self.layers[i](x[:, i]) for i in range(x.size(1))], 
            dim=1,
        ).squeeze(-1)
        if reduction:
            x = x.sum(-1)
        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(
            self, 
            n_embeddings, 
            embedding_dim, 
            commitment_cost=0.25, 
            decay=0.999,
            epsilon=1e-5,
        ):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach(), reduction="none")
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, output_dim, cfg: Wav2vecU_GraphConfig):
        super().__init__()

        self.cfg = cfg
        self.codebook_size = cfg.quantizer_codebook_size
        self.decode_obj = cfg.quantizer_decode_objective
        self.codebook = VQEmbeddingEMA(self.codebook_size, output_dim)
        if self.decode_obj == "cross_entropy":
            self.decoder = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 512, bias=False),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, cfg.quantizer_codebook_size, bias=False),
            )
        else:
            self.decoder = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Linear(512, 512, bias=False),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, output_dim, bias=False),
            )

    def forward(self, dense_x, dense_padding_mask, labels):
        quantized, vq_loss, perplexity = self.codebook(dense_x)
        out = self.decoder(dense_x)
        labels[dense_padding_mask] = -1
        if self.decode_obj == "cross_entropy":
            recon_loss = F.cross_entropy(
                out.transpose(1, 2), labels, 
                ignore_index=-1, 
                reduction="none",
            )
        else:
            recon_loss = F.mse_loss(
                out, labels,
                reduction="none",
            )
        loss = recon_loss + vq_loss.sum(-1)
        result = {
            "quantized": quantized,
            "dense_padding_mask": dense_padding_mask,
            "loss": loss,
        }
        return result

    def encode(self, dense_x, dense_padding_mask):
        quantized, indices = self.codebook.encode(dense_x)
        result = {
            "quantized": quantized,
            "dense_padding_mask": dense_padding_mask,
            "indices": indices,
        }
        return result 
   

class GumbelQuantizer(nn.Module):
    def __init__(self, input_dim, cfg: Wav2vecU_GraphConfig):
        super().__init__()

        self.cfg = cfg
        self.codebook_size = cfg.quantizer_codebook_size
        self.hidden_dim = 1024
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, cfg.quantizer_codebook_size),
        )

    def forward(self, dense_x, dense_padding_mask, labels=None):
        logits = self.encoder(dense_x)
        #h_0 = dense_x.new_zeros(2*self.num_layers, dense_x.size(0), self.hidden_dim)
        #c_0 = h_0
        #logits = self.classifier(self.encoder(dense_x, (h_0, c_0))[0])
        #quantized = F.gumbel_softmax(logits, tau=1, hard=True) 
        quantized = torch.softmax(logits, dim=-1) 
        result = {
            "quantized": quantized,
            "dense_padding_mask": dense_padding_mask,
            "indices": quantized.argmax(-1),
        }
        if labels is None:
            return result

        labels[dense_padding_mask] = -1
        loss = F.cross_entropy(
            logits.transpose(1, 2), labels,
            ignore_index=-1,
            reduction="none",
        )
        result["loss"] = loss 
        return result

    def encode(self, dense_x, dense_padding_mask):
        return self(dense_x, dense_padding_mask)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: Wav2vecU_GraphConfig):
        super().__init__()

        self.cfg = cfg
        self.input_type = cfg.generator_input_type
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)
        self.batch_norm = cfg.generator_batch_norm != 0
        self.residual = cfg.generator_residual
        self.avg_pool = cfg.generator_avg_pool_kernel > 0
        self.avg_pool_stride = cfg.generator_avg_pool_stride
        if cfg.generator_input_type == "int":
            self.proj = nn.Embedding(
                input_dim,
                output_dim,
            )
        else:
            self.proj = nn.Sequential(
                TransposeLast(),
                nn.Conv1d(
                    input_dim,
                    output_dim,
                    kernel_size=cfg.generator_kernel,
                    stride=cfg.generator_stride,
                    dilation=cfg.generator_dilation,
                    padding="same",
                    bias=cfg.generator_bias,
                ),
                TransposeLast(),
            )

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(cfg.generator_batch_norm)
        if self.residual:
            self.in_proj = nn.Linear(input_dim, input_dim)
        if self.avg_pool:
            avg_pool_padding = (
                cfg.generator_avg_pool_kernel // 2 
            )
            self.pool = nn.AvgPool1d(
                kernel_size=cfg.generator_avg_pool_kernel,
                stride=cfg.generator_avg_pool_stride,
                padding=avg_pool_padding,
            )
    
    def forward(self, dense_x, tokens, dense_padding_mask):
        result = {}
        if self.input_type == "int":
            dense_x = dense_x.long().squeeze(-1)
        else:
            if self.batch_norm:
                dense_x = self.bn_padded_data(dense_x, dense_padding_mask)
            if self.residual:
                inter_x = self.in_proj(self.dropout(dense_x))
                dense_x = dense_x + inter_x
                result["inter_x"] = inter_x
            dense_x = self.dropout(dense_x)
        dense_x = self.proj(dense_x)
        
        if self.avg_pool:
            dense_x = self.pool(
                dense_x.permute(0, 2, 1)
            ).permute(0, 2, 1)
        
        if self.stride * self.avg_pool_stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride*self.avg_pool_stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))
        
        # Compute dense_x and token_x co-occurence matrices
        result["dense_x"] = dense_x
        result["token_x"] = token_x
        result["dense_padding_mask"] = dense_padding_mask

        return result

    def bn_padded_data(self, feature, padding_mask):
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature


@register_model("wav2vecu_graph", dataclass=Wav2vecU_GraphConfig)
class Wav2vecU_Graph(BaseFairseqModel):
    def calc_gradient_penalty(self, real_data, fake_data):

        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        if self.cfg.probabilistic_grad_penalty_slicing:

            def get_slice(data, dim, target_size):

                size = data.size(dim)
                diff = size - target_size
                if diff <= 0:
                    return data

                start = np.random.randint(0, diff + 1)
                return data.narrow(dim=dim, start=start, length=target_size)

            real_data = get_slice(real_data, 0, b_size)
            real_data = get_slice(real_data, 1, t_size)
            fake_data = get_slice(fake_data, 0, b_size)
            fake_data = get_slice(fake_data, 1, t_size)

        else:
            real_data = real_data[:b_size, :t_size]
            fake_data = fake_data[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self.discriminator(interpolates, None)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def quantize_step(self, num_updates):
        return False #num_updates < 300

    def segment_step(self, num_updates):
        return True

    def join_segment_step(self, num_updates):
        return num_updates > 200

    def sample_skipsizes(self, k, num_updates):
        if k == 0:
            return []
        skipsizes = [(i, j) for i in range(1, k+1) for j in range(1, k+1)]
        i = num_updates % k
        return [skipsizes[(i+j)%k] for j in range(len(skipsizes))]
        # return skipsizes

    def discrim_step(self, num_updates):
        if self.gan_type == "L1":
            return False
        return num_updates % 2 == 0

    def get_groups_for_update(self, num_updates):
        if self.quantize_step(num_updates):
            return {"quantizer"}
        elif self.segment_step(num_updates):
            return {"generator", "quantizer"}
        else:
            return {"discriminator"} if self.discrim_step(num_updates) else {"generator"} 

    def __init__(self, cfg: Wav2vecU_GraphConfig, target_dict):
        super().__init__()

        self.cfg = cfg
        self.gan_type = cfg.gan_type
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
        self.nspecial = target_dict.nspecial
        self.no_special_tokens = cfg.no_special_tokens
        self.no_silence = cfg.no_silence

        self.skip_size = cfg.skipgram_size
        self.skipgram_only = cfg.skipgram_only
        self.position_skipgram = cfg.position_skipgram
        self.tri_size = cfg.trigram_size

        self.smoothness_weight = cfg.smoothness_weight
        self.segment_weight = cfg.segment_weight
        self.quantizer_weight = cfg.quantizer_weight

        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.smoothing = cfg.smoothing
        self.smoothing_one_sided = cfg.smoothing_one_sided
        self.no_softmax = cfg.no_softmax
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel
        self.last_acc = None

        self.gradient_penalty = cfg.gradient_penalty
        self.code_penalty = cfg.code_penalty
        self.mmi_weight = cfg.mmi_weight
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
        assert self.blank_index != target_dict.unk()
        
        if self.gan_type == "L1":
            self.discriminator = None
            self.skip_discriminator = None
        else:
            if cfg.discriminator_type == "mlp":
                self.discriminator = MLPDiscriminator(output_size, cfg)
                self.skip_discriminator = MLPDiscriminator(output_size, cfg)
            else:
                self.discriminator = Discriminator(output_size, cfg)
                self.skip_discriminator = Discriminator(output_size, cfg)
            self.reset_discriminator_every_update = cfg.reset_discriminator_every_update
            
            for p in self.discriminator.parameters():
                p.param_group = "discriminator"
            for p in self.skip_discriminator.parameters():
                p.param_group = "discriminator"

        self.pca_A = self.pca_b = None
        d = cfg.input_dim

        # self.cfg.segmentation.type = SegmentationType.BINARY
        self.segmenter = SEGMENT_FACTORY[cfg.segmentation.type](cfg.segmentation)
        self.join_segmenter = JoinSegmenter(cfg.segmentation)

        self.generator = Generator(d, output_size, cfg)

        self.quantizer = None
        if cfg.quantizer_codebook_size > 0:
            if cfg.quantizer_type == "vq":
                self.quantizer = VectorQuantizer(cfg.quantizer_input_dim, cfg)
            else:
                self.quantizer = GumbelQuantizer(cfg.quantizer_input_dim, cfg) 
            for p in self.quantizer.parameters():
                p.param_group = "quantizer"  #"generator"

        for p in self.generator.parameters():
            p.param_group = "generator"
        
        for p in self.segmenter.parameters():
            p.param_group = "generator"

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0

        if self.mmi_weight > 0:
            self.target_downsample_rate = cfg.target_downsample_rate
            self.decoder = nn.Linear(d, cfg.target_dim)
            for p in self.decoder.parameters():
                p.param_group = "generator"

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.target_dictionary)

    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output["logits"]

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., self.blank_index] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., self.blank_index] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        padding = net_output["padding_mask"]
        if padding.any():
            logits[padding] = float("-inf")
            logits[padding][..., self.blank_index] = float("inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)

    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)

        probs = super().get_normalized_probs(logits, log_probs, sample)
        # BTC -> TBC for ctc
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )

        if not self.no_softmax:
            if self.training and self.gumbel:
                dense_x = F.gumbel_softmax(
                    dense_x.float(), tau=self.curr_temp, hard=self.hard_gumbel
                ).type_as(dense_x)
            else:
                dense_x = dense_x.softmax(-1)

        return dense_x, code_perplexity, prob_perplexity
    
    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        token_x=None,
        dense_x_only=False,
        segment=True,
        aux_target=None,
        clus_features=None,
        bin_labels=None,
        gt_bin_labels=None,
    ):
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)

        orig_size = features.size(0) * features.size(1) - padding_mask.sum()

        quantizer_loss = None
        if self.quantizer is None:
            if clus_features is not None:
                gen_result = self.generator(clus_features, random_label, padding_mask)
                orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
            else:
                gen_result = self.generator(features, random_label, padding_mask)
                orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
        else:
            if dense_x_only:
                q_result = self.quantizer.encode(features, padding_mask)
                quantized = q_result["quantized"]
            else:
                clus_labels = clus_features.argmax(-1).long()
                q_result = self.quantizer(
                    features, padding_mask, labels=clus_labels,
                )
                quantized = q_result["quantized"]
                q_loss = q_result["loss"]               
                if self.quantizer_weight > 0:
                    quantizer_loss = q_loss.sum() * self.quantizer_weight
 
            gen_result = self.generator(quantized, random_label, padding_mask)
            orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
       
        orig_dense_padding_mask = gen_result["dense_padding_mask"]
        if segment:
            if self.cfg.segmentation.type == SegmentationType.BINARY:
                dense_x, dense_padding_mask = self.segmenter.logit_segment(
                    (orig_dense_x, features), orig_dense_padding_mask,
                )
            else:
                dense_x, dense_padding_mask = self.segmenter.logit_segment(
                    orig_dense_x, orig_dense_padding_mask,
                )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        if self.join_segment_step(self.update_num):
            dense_x, dense_padding_mask = self.join_segmenter.logit_segment(
                orig_dense_x, orig_dense_padding_mask,
            )

        dense_logits = dense_x
        if self.no_silence:
            dense_logits[..., self.zero_index] = -1e14
        if self.no_special_tokens:
            dense_logits[..., :self.nspecial] = -1e14
 
        prob_perplexity = None
        code_perplexity = None

        if not (self.no_softmax and dense_x_only):
            dense_x, code_perplexity, prob_perplexity = self.normalize(dense_logits)

        if dense_x_only:
            result = {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }
            if self.cfg.segmentation.type == SegmentationType.BINARY:
                bin_scores = self.segmenter(
                    (orig_dense_x, features), orig_dense_padding_mask
                )
                result["bin_scores"] = bin_scores
            if self.quantizer is not None:
                result["quantized_indices"] = q_result["indices"]
            return result
             
        token_padding_mask = random_label == self.pad

        bsz, tsz, dsz = dense_x.size()
        dense_x = dense_x * (1 - dense_padding_mask.unsqueeze(-1).float())
        try:
            if dense_x.size(1) > token_x.size(1):
                gap = dense_x.size(1) - token_x.size(1)
                pad = dense_x.new_zeros(bsz, gap, dsz)
                token_x = torch.cat((token_x, pad), dim=1)
                pad[:, :, self.pad] = 1.0
                token_padding_mask = torch.cat(
                    (
                        token_padding_mask, 
                        dense_x.new_ones(bsz, gap).bool(),
                    ), dim=1,
                )
            elif dense_x.size(1) < token_x.size(1):
                gap = token_x.size(1) - dense_x.size(1)
                pad = dense_x.new_zeros(bsz, gap, dsz)
                dense_logits = torch.cat((dense_logits, pad), dim=1)
                pad[:, :, self.pad] = 1.0
                dense_x = torch.cat((dense_x, pad), dim=1)
                dense_padding_mask = torch.cat(
                    (
                        dense_padding_mask, 
                        dense_x.new_ones(bsz, gap).bool(),
                    ), dim=1,
                )
        except:
            random_label = dense_x.new_zeros(
                dense_x.size(0), dense_x.size(1),
            )
            token_x = dense_x.new_zeros(*dense_x.size())
            dense_padding_mask = dense_padding_mask.new_ones(
                *dense_padding_mask.size(),
            )
            token_padding_mask = dense_padding_mask
        skip_dense_x = []
        for skip in range(1, self.skip_size+1):
            if self.position_skipgram:
                count_dense_x = torch.matmul(
                    dense_x[:, :-skip].permute(1, 2, 0), 
                    dense_x[:, skip:].permute(1, 0, 2),
                )
            else:
                count_dense_x = torch.mm(
                    dense_x[:, :-skip].reshape(-1, dsz).t(), 
                    dense_x[:, skip:].reshape(-1, dsz),
                )
            skip_dense_x.append(count_dense_x)
        if self.skip_size > 0:
            skip_dense_x = torch.cat(skip_dense_x).unsqueeze(0)
            skip_dense_padding_mask = dense_x.new_zeros(
                skip_dense_x.size()[:-1],
            )

        # Trigram
        tri_dense_x = []
        for skip1, skip2 in self.sample_skipsizes(self.tri_size, self.update_num):
            # (B x (T - S), D, D)
            count_dense_x = dense_x[:, :-skip1-skip2].reshape(-1, dsz, 1) * dense_x[:, skip1:-skip2].reshape(-1, 1, dsz)

            # (D, D, D)
            count_dense_x = torch.matmul(
                count_dense_x.permute(1, 2, 0),
                dense_x[:, skip1+skip2:].reshape(-1, dsz).unsqueeze(0),
            )
            tri_dense_x.append(count_dense_x)
        if self.tri_size > 0:
            tri_dense_x = torch.cat(tri_dense_x).unsqueeze(0)
            tri_dense_padding_mask = dense_x.new_zeros(
                tri_dense_x.size()[:-1],
            )

        bsz, tsz = random_label.size()
        token_x = token_x * (1 - token_padding_mask.unsqueeze(-1).float())
        skip_token_x = []
        for skip in range(1, self.skip_size+1): 
            if self.position_skipgram:
                count_token_x = torch.matmul(
                    token_x[:, :-skip].permute(1, 2, 0),
                    token_x[:, skip:].permute(1, 0, 2),
                )
            else:
                count_token_x = torch.mm(
                    token_x[:, :-skip].reshape(-1, dsz).t(),
                    token_x[:, skip:].reshape(-1, dsz),
                )
            skip_token_x.append(count_token_x) 
        if self.skip_size > 0:
            skip_token_x = torch.cat(skip_token_x).unsqueeze(0)       
            skip_token_padding_mask = token_x.new_zeros(
                skip_token_x.size()[:-1],
            )

        tri_token_x = []
        for skip1, skip2 in self.sample_skipsizes(self.tri_size, self.update_num):
            # (B x (T - S), D, D)
            count_token_x = token_x[:, :-skip1-skip2].reshape(-1, dsz, 1) * token_x[:, skip1:-skip2].reshape(-1, 1, dsz)

            # (D, D, D)
            count_token_x = torch.matmul(
                count_token_x.permute(1, 2, 0),
                token_x[:, skip1+skip2:].reshape(-1, dsz).unsqueeze(0),
            )
            tri_token_x.append(count_token_x)
        if self.tri_size > 0:
            tri_token_x = torch.cat(tri_token_x).unsqueeze(0)
            tri_token_padding_mask = token_x.new_zeros(
                tri_token_x.size()[:-1],
            )

        q_step = self.quantize_step(self.update_num)
        s_step = self.segment_step(self.update_num)
        d_step = self.discrim_step(self.update_num)
        if d_step and self.reset_discriminator_every_update:
            if isinstance(self.discriminator, MLPDiscriminator) and self.discriminator.depth > 0:
                for p in self.discriminator.parameters():
                    if (p.ndim == 2) and (p.size(0) == 1):
                        nn.init.xavier_normal_(p)
                    else:
                        p.data.fill_(0.0)
                for p in self.skip_discriminator.parameters():
                    if (p.ndim == 2) and (p.size(0) == 1):
                        nn.init.xavier_normal_(p)
                    else:
                        p.data.fill_(0.0)
            else:
                for p in self.discriminator.parameters():
                    p.data.fill_(0.0)
                for p in self.skip_discriminator.parameters():
                    p.data.fill_(0.0)
            
        sample_size = dense_x.size(0)

        fake_smooth = self.smoothing
        real_smooth = self.smoothing
        if self.smoothing_one_sided:
            fake_smooth = 0

        zero_loss = None
        smoothness_loss = None
        code_pen = None
        mmi_loss = None
        segment_loss = None
        grad_pen = None
        loss_token = None
        loss_dense = None
        if self.gan_type != "L1":
            dense_y = self.discriminator(dense_x, dense_padding_mask > 0)
            token_y = self.discriminator(token_x, token_padding_mask > 0)
        if d_step:
            if self.gan_type == "MMD":
                loss_dense = torch.mean(dense_y) * sample_size
                loss_token = - torch.mean(token_y) * sample_size
                if self.skip_size > 0:
                    loss_dense += torch.mean(skip_dense_y) * sample_size
                    loss_token += - torch.mean(skip_token_y) * sample_size
            elif self.gan_type == "JSD":
                loss_dense = F.binary_cross_entropy_with_logits(
                    dense_y,
                    dense_y.new_ones(dense_y.shape) - fake_smooth,
                    reduction="sum",
                )
                loss_token = F.binary_cross_entropy_with_logits(
                    token_y,
                    token_y.new_zeros(token_y.shape) + real_smooth,
                    reduction="sum",
                )
                if self.skip_size > 0:
                    loss_dense += F.binary_cross_entropy_with_logits(
                        skip_dense_y,
                        skip_dense_y.new_ones(skip_dense_y.shape) - fake_smooth,
                        reduction="sum",
                    )
                    loss_token += F.binary_cross_entropy_with_logits(
                        skip_token_y,
                        skip_token_y.new_zeros(skip_token_y.shape) + real_smooth,
                        reduction="sum",
                    )
            else:
                raise ValueError(f"Unknown GAN type: {self.gan_type}")

            if self.training and self.gradient_penalty > 0:
                grad_pen = self.calc_gradient_penalty(token_x, dense_x)
                grad_pen = grad_pen.sum() * self.gradient_penalty
            else:
                grad_pen = None
        elif not q_step:           
            if self.gan_type == "L1":
                loss_dense = F.l1_loss(dense_x.sum(0), token_x.sum(0), reduction="sum")
                if self.skip_size > 0:
                    loss_dense += F.l1_loss(skip_dense_x, skip_token_x, reduction="sum")
                if self.tri_size > 0:
                    loss_dense += F.l1_loss(tri_dense_x, tri_token_x, reduction="sum")
            elif self.gan_type == "MMD":
                loss_dense = (torch.mean(token_y) - torch.mean(dense_y)) * sample_size
                if self.skip_size > 0:
                    loss_dense += (torch.mean(skip_token_y) - torch.mean(skip_dense_y)) * sample_size
            elif self.gan_type == "JSD":
                loss_dense = F.binary_cross_entropy_with_logits(
                    dense_y,
                    dense_y.new_zeros(dense_y.shape) + fake_smooth,
                    reduction="sum",
                )
                if self.skip_size > 0:
                    loss_dense += F.binary_cross_entropy_with_logits(
                        skip_dense_y,
                        skip_dense_y.new_zeros(skip_dense_y.shape) + fake_smooth,
                        reduction="sum",
                    )
            else:
                raise ValueError(f"Unknown GAN type: {self.gan_type}")

            if s_step and self.segment_weight > 0:
                if self.cfg.segmentation.type == SegmentationType.BINARY:
                    _, segment_loss = self.segmenter(
                        (orig_dense_x, features), orig_dense_padding_mask, labels=bin_labels,
                    )
                else:
                    _, segment_loss = self.segmenter(
                        orig_dense_x, orig_dense_padding_mask,
                    )
                segment_loss = segment_loss.sum() * self.segment_weight

            num_vars = dense_x.size(-1)
            if prob_perplexity is not None:
                code_pen = (num_vars - prob_perplexity) / num_vars
                code_pen = code_pen * sample_size * self.code_penalty

            if self.smoothness_weight > 0:
                smoothness_loss = F.mse_loss(
                    dense_logits[:, :-1], dense_logits[:, 1:], reduction="none"
                )
                smoothness_loss[dense_padding_mask[:, 1:]] = 0
                smoothness_loss = (
                    smoothness_loss.mean() * sample_size * self.smoothness_weight
                )

            if (self.mmi_weight > 0) and (aux_target is not None):
                inter_x = self.decoder(gen_result["inter_x"])
                if self.target_downsample_rate > 1:
                    aux_target = aux_target[:, :: self.target_downsample_rate]
                max_t_len = min(aux_target.shape[1], inter_x.shape[1])
                mmi_loss = F.cross_entropy(
                    inter_x[:, :max_t_len].transpose(1, 2),
                    aux_target[:, :max_t_len],
                    ignore_index=-1,
                    reduction="none",
                )
                mmi_loss = mmi_loss.mean() * mmi_loss.shape[0] * self.mmi_weight

        result = {
            "losses": {
                "grad_pen": grad_pen,
                "code_pen": code_pen,
                "smoothness": smoothness_loss,
                "mmi": mmi_loss,
                "segment_loss": segment_loss,
                "quantizer_loss": quantizer_loss,
            },
            "temp": self.curr_temp,
            "code_ppl": code_perplexity,
            "prob_ppl": prob_perplexity,
            "d_steps": int(d_step),
            "sample_size": sample_size,
        }

        suff = "_d" if d_step else "_g"
        result["losses"]["dense" + suff] = loss_dense
        result["losses"]["token" + suff] = loss_token
        return result
