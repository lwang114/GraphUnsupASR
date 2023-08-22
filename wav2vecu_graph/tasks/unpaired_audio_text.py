# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from dataclasses import dataclass, field
import logging
import math
import numpy as np
import os
from typing import Optional
import torch

from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from ..data import ExtractedFeaturesDataset, RandomInputDataset

from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed.utils import get_data_parallel_world_size
from omegaconf import MISSING

from examples.speech_recognition.kaldi.kaldi_decoder import (
    KaldiDecoder,
    KaldiDecoderConfig,
)

from dtw import *
import json
from pathlib import Path
import pdb


logger = logging.getLogger(__name__)


@dataclass
class DecodingConfig(FairseqDataclass):
    kenlm_path: Optional[str] = None
    lm_weight: float = 0
    blank_weight: float = 0


@dataclass
class UnpairedAudioTextConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing audio"}
    )
    text_data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing text"}
    )
    segment_data: Optional[str] = field(
        default="", 
        metadata={
            "help": "path to data directory containing segmentation"
        },
    )
    max_length: Optional[int] = None
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    #aux_target_postfix: Optional[str] = field(
    #    default=None,
    #    metadata={"help": "auxaliry target filename extension"},
    #)
    unfiltered: bool = field(
        default=False, metadata={"help": "load data with _unfiltered suffix"}
    )
    ctc_eval: bool = field(
        default=False, metadata={"help": "eval UER as if computed by CTC"}
    )
    sort_by_length: bool = field(
        default=True, metadata={"help": "sort examples by length of audio timesteps"}
    )
    shuffle: bool = field(default=True, metadata={"help": "shuffle examples"})
    append_eos: bool = field(default=False, metadata={"help": "append eos"})
    uppercase: Optional[bool] = field(
        default=False, metadata={"help": "uppercase for LM score computation"}
    )
    skipwords: Optional[str] = field(
        default="",
        metadata={
            "help": "comma-separated words to be removed for LM score computation"
        },
    )
    kenlm_path: Optional[str] = None
    vocab_usage_power: float = 2
    random_choice: bool = field(default=True, metadata={"help": "use random choice for sampling unpaired data"})

    word_decoder_config: Optional[KaldiDecoderConfig] = None
    word_kenlm_path: Optional[str] = None

    decoding_config: DecodingConfig = DecodingConfig()


@register_task("unpaired_audio_text", dataclass=UnpairedAudioTextConfig)
class UnpairedAudioText(FairseqTask):
    """ """

    cfg: UnpairedAudioTextConfig

    def __init__(
        self,
        cfg: UnpairedAudioTextConfig,
        source_dictionary=None,
        target_dictionary=None,
    ):
        super().__init__(cfg)

        self.tolerance = 1
        self._target_dictionary = target_dictionary
        self._source_dictionary = source_dictionary
        self.num_symbols = (
            len([s for s in target_dictionary.symbols if not s.startswith("madeup")])
            - target_dictionary.nspecial
        )
        self.sil_id = (
            target_dictionary.index("<SIL>") if "<SIL>" in target_dictionary else -1
        )
        self.kenlm = None
        if cfg.kenlm_path is not None:
            import kenlm

            self.kenlm = kenlm.Model(cfg.kenlm_path)

        self.word_kenlm = None
        if cfg.word_kenlm_path is not None:
            import kenlm

            self.word_kenlm = kenlm.Model(cfg.word_kenlm_path)

        self.uppercase = cfg.uppercase
        self.skipwords = set(cfg.skipwords.split(","))

        def str_postprocess(s):
            s = " ".join(w for w in s.split() if w not in self.skipwords)
            s = s.upper() if self.uppercase else s
            return s

        self.str_postprocess = str_postprocess
        self.compute_lm_score = lambda s: self.kenlm.score(self.str_postprocess(s))

        self.compute_word_score = None
        if cfg.word_decoder_config is not None:
            self.kaldi_decoder = KaldiDecoder(cfg.word_decoder_config, beam=10)

            def compute_word_score(logits, padding):
                res = self.kaldi_decoder.decode(logits, padding)
                for r in res:
                    r = r.result()
                    assert len(r) == 1
                    r = r[0]
                    yield r["score"], r["words"]

            self.compute_word_score = compute_word_score

    @classmethod
    def setup_task(cls, cfg: UnpairedAudioTextConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        dict_path = os.path.join(cfg.text_data, "dict.txt")
        if os.path.exists(dict_path):
            target_dictionary = Dictionary.load(dict_path)
        else:
            dict_path = os.path.join(cfg.data, f"dict.{cfg.labels}.txt")
            target_dictionary = Dictionary.load(dict_path)

        return cls(cfg, target_dictionary=target_dictionary)

    def optimizer_step(self, optimizer, model, update_num):
        if hasattr(model, "get_groups_for_update"):
            groups = model.get_groups_for_update(update_num)
            optimizer.step(groups=groups)
        else:
            optimizer.step()

    def valid_step(self, sample, model, criterion):
        res = model(
            **sample["net_input"],
            dense_x_only=True,
        )

        dense_x = res["logits"]
        padding_mask = res["padding_mask"]
        bin_scores = res["bin_scores"] if "bin_scores" in res else [None] * len(dense_x)
        quantized_indices = res["quantized_indices"] if "quantized_indices" in res else [None] * len(dense_x)

        word_scores = None
        if self.compute_word_score is not None:
            word_scores = self.compute_word_score(dense_x.cpu(), padding_mask.cpu())

        z = dense_x.argmax(-1)
        z[padding_mask] = self.target_dictionary.pad()

        vocab_seen = torch.zeros(self.num_symbols, dtype=torch.bool)

        import editdistance

        c_err = 0
        c_len = 0
        pred_c_len = 0
        lm_score_sum = 0
        b_len = 0
        gt_b_len = 0
        pred_b_len = 0

        p_count = 0
        r_count = 0
        p_dup_count = 0
        r_dup_count = 0
        gt_p_count = 0
        gt_r_count = 0
        gt_p_dup_count = 0
        gt_r_dup_count = 0

        q_err = 0
        q_len = 0
        for i, (x, t, id, bin_label, gt_bin_label, bin_score, q_indices, clus_feature) in enumerate(
            zip(
                z,
                sample["target"] if "target" in sample else [None] * len(z),
                sample["id"],
                sample["net_input"]["bin_labels"] if sample["net_input"]["bin_labels"] is not None else [None] * len(z),
                sample["net_input"]["gt_bin_labels"] if sample["net_input"]["gt_bin_labels"] is not None else [None] * len(z),
                bin_scores,
                quantized_indices,
                sample["net_input"]["clus_features"] if sample["net_input"]["clus_features"] is not None else [None] * len(z)
            )
        ):  
            if t is not None:
                t = t[(t >= self.target_dictionary.nspecial)]
            x = x[
                (x >= self.target_dictionary.nspecial)
                & (x < (self.num_symbols + self.target_dictionary.nspecial))
            ]
            if self.sil_id >= 0:
                x = x[x != self.sil_id]

            vocab_seen[x - self.target_dictionary.nspecial] = True

            pred_units_arr = x
            if self.cfg.ctc_eval:
                pred_units_arr = pred_units_arr.unique_consecutive()
                pred_units_arr = pred_units_arr[pred_units_arr != 0]

            y = None
            if bin_label is not None:
                y = (bin_label == 1).nonzero().squeeze(-1).cpu().numpy()
            
            y_gt = None
            if gt_bin_label is not None:
                y_gt = gt_bin_label.nonzero().squeeze(-1).cpu().numpy()

            yhat = None
            if bin_score is not None:
                pred_bin_label = (bin_score > 0).long() 
                yhat = pred_bin_label.nonzero().squeeze(-1).cpu().numpy()

            clus_label = None
            if clus_feature is not None:
                clus_label = clus_feature.argmax(-1)
                clus_label = clus_label[clus_feature.sum(-1) > 0]

            if q_indices is not None:
                q_indices = q_indices[clus_feature.sum(-1) > 0]
                q_err += (q_indices != clus_label).long().sum().item()
                q_len += len(clus_label)
                
            if id == 0:
                if t is not None:
                    logger.info(f"REF: {self.target_dictionary.string(t)}")
                logger.info(f"HYP: {self.target_dictionary.string(pred_units_arr)}")
                if gt_bin_label is not None:
                    logger.info(f"GT boundaries: {y_gt.tolist()}")
                
                if bin_label is not None:
                    logger.info(f"REF boundaries: {y.tolist()}")

                if bin_score is not None:
                    logger.info(f"HYP boundaries: {yhat.tolist()}")

                if clus_label is not None:
                    logger.info(f"CLUS REF: {clus_label.tolist()}")
                    
                if q_indices is not None:
                    logger.info(f"CLUS HYP: {q_indices.tolist()}")

                if self.kenlm is not None:
                    if t is not None:
                        ref_lm_s = self.compute_lm_score(
                            self.target_dictionary.string(t)
                        )
                        logger.info(
                            f"LM [REF]: {ref_lm_s}, {math.pow(10, -ref_lm_s / (len(t) + 1))}"
                        )

                    hyp_lm_s = self.compute_lm_score(
                        self.target_dictionary.string(pred_units_arr)
                    )
                    logger.info(
                        f"LM [HYP]: {hyp_lm_s}, {math.pow(10, -hyp_lm_s / (len(pred_units_arr) + 1))}"
                    )

            pred_units_arr = pred_units_arr.tolist()

            pred_c_len += len(pred_units_arr)

            if t is not None:
                t = t.tolist()
                c_err += editdistance.eval(pred_units_arr, t)
                c_len += len(t)
            else:
                c_len = pred_c_len

            if y is not None and yhat is not None:
                b_len += len(y)
                pred_b_len += len(yhat)
                p, pd = self.get_counts(y, yhat)
                p_count += p
                p_dup_count += pd
                r, rd = self.get_counts(yhat, y)
                r_count += r
                r_dup_count += rd

            if y_gt is not None and yhat is not None:
                gt_b_len += len(y_gt)
                p, pd = self.get_counts(y_gt, yhat)
                gt_p_count += p
                gt_p_dup_count += pd
                r, rd = self.get_counts(yhat, y_gt)
                gt_r_count += r
                gt_r_dup_count += rd

            if self.kenlm is not None:
                pred_str = self.target_dictionary.string(pred_units_arr)
                lm_score = self.compute_lm_score(pred_str)
                lm_score_sum += lm_score

        kaldi_score_sum = 0
        word_lm_sum = 0
        num_words = 0
        if word_scores is not None:
            for score, words in word_scores:
                kaldi_score_sum += score
                num_words += len(words)
                if self.word_kenlm is not None:
                    word_lm_sum += self.kenlm.score(" ".join(words))
        try:
            world_size = get_data_parallel_world_size()
        except:
            world_size = 1

        tokens = sample["target"]
        token_x = dense_x.new_zeros(tokens.numel(), self.num_symbols+self.target_dictionary.nspecial)  
        token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
        token_x = token_x.view(tokens.shape + (self.num_symbols+self.target_dictionary.nspecial,))

        logging_output = {
            "loss": c_err,
            "_num_char_errors": c_err,
            "_num_chars": c_len,
            "_num_pred_chars": pred_c_len,
            "_num_precision_counts": p_count,
            "_num_recall_counts": r_count,
            "_num_segments": b_len,
            "_num_gt_precision_counts": gt_p_count,
            "_num_gt_recall_counts": gt_r_count,
            "_num_gt_segments": gt_b_len,
            "_num_pred_segments": pred_b_len,
            "_num_quantize_errors": q_err,
            "_num_quantize_units": q_len,
            "ntokens": c_len,
            "nsentences": z.size(0),
            "sample_size": c_len,
            "_world_size": world_size,
            "_lm_score_sum": lm_score_sum,
            "_kaldi_score_sum": kaldi_score_sum,
            "_word_lm_sum": word_lm_sum,
            "_num_words": num_words,
            "_vocab_seen": vocab_seen,
        }

        return c_err, c_len, logging_output

    def get_assignments(self, y, yhat):
        matches = dict((i, []) for i in range(len(yhat)))
        for i, yhat_i in enumerate(yhat):
            dists = np.abs(y - yhat_i)
            idxs = np.argsort(dists)
            for idx in idxs:
                if dists[idx] <= self.tolerance:
                    matches[i].append(idx)
        return matches

    def get_counts(self, y, yhat):
        match_counter = 0
        dup_counter = 0
        miss_counter = 0
        used_idxs = []
        matches = self.get_assignments(y, yhat)
        dup_frames = []
        miss_frames = []

        for m, vs in matches.items():
            if len(vs) == 0:
                miss_frames.append(m)
                miss_counter += 1
                continue
            vs = sorted(vs)
            dup = False
            for v in vs:
                if v in used_idxs:
                    dup = True
                else:
                    dup = False
                    used_idxs.append(v)
                    match_counter += 1
                    break
            if dup:
                dup_counter += 1
                dup_frames.append(m)

        return match_counter, dup_counter

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        segment_path = self.cfg.segment_data
        task_cfg = task_cfg or self.cfg

        has_unpaired_text = os.path.exists(
            os.path.join(self.cfg.text_data, f"{split}.idx")
        )

        self.datasets[split] = ExtractedFeaturesDataset(
            path=data_path,
            split=split,
            min_length=3,
            max_length=task_cfg.max_length,
            labels=None if has_unpaired_text else task_cfg.labels,
            label_dict=self.target_dictionary,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
            segment_path=segment_path,
            # aux_target_postfix=task_cfg.aux_target_postfix,
        )

        logger.info(f"split {split} has unpaired text? {has_unpaired_text}")
        if has_unpaired_text:
            text_dataset = data_utils.load_indexed_dataset(
                os.path.join(self.cfg.text_data, split), self.target_dictionary
            )
            text_dataset = StripTokenDataset(text_dataset, self.target_dictionary.eos())
            self.datasets[split] = RandomInputDataset(
                self.datasets[split],
                text_dataset,
                ["random_label"],
                add_to_input=True,
                pad_idx=self.target_dictionary.pad(),
                random_choice=self.cfg.random_choice,
            )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def similarity(self, src_embs, tgt_embs, src_lens, tgt_lens):
        assert src_embs.dim() == tgt_embs.dim() == 3
        n = src_embs.size(0)
        S = src_embs.new_zeros(n, n)
        dist_mats = []
        alignments = []
        for src_idx, (src_emb, src_len) in enumerate(zip(src_embs, src_lens)):
            alignments.append([])
            dist_mats.append([])
            for tgt_idx, (tgt_emb, tgt_len) in enumerate(zip(tgt_embs, tgt_lens)):
                if src_len <= 0 or tgt_len <= 0:
                    continue
                dist_mat = - torch.mm(src_emb, tgt_emb.t())
                dist_mat = dist_mat[:src_len, :tgt_len]
                alignment = dtw(
                    dist_mat.cpu().numpy().astype("double")
                )
                min_dist = torch.tensor(
                    alignment.distance
                )
                dist_mats[-1].append(
                    dist_mat.cpu().tolist()
                )
                alignments[-1].append(
                    [
                        alignment.index1.tolist(), 
                        alignment.index2.tolist(),
                    ]
                )
                S[src_idx, tgt_idx] = - min_dist
        return S, dist_mats, alignments

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        num_pred_chars = sum(
            log.get("_num_pred_chars", zero) for log in logging_outputs
        )

        num_precision_counts = sum(
            log.get("_num_precision_counts", zero) for log in logging_outputs
        )
        num_recall_counts = sum(
            log.get("_num_recall_counts", zero) for log in logging_outputs
        )
        num_segments = sum(
            log.get("_num_segments", zero) for log in logging_outputs
        )
        num_gt_precision_counts = sum(
            log.get("_num_gt_precision_counts", zero) for log in logging_outputs
        )
        num_gt_recall_counts = sum(
            log.get("_num_gt_recall_counts", zero) for log in logging_outputs
        )
        num_gt_segments = sum(
            log.get("_num_gt_segments", zero) for log in logging_outputs
        )
        num_pred_segments = sum(
            log.get("_num_pred_segments", zero) for log in logging_outputs
        )

        num_quantize_errors = sum(
            log.get("_num_quantize_errors", zero) for log in logging_outputs
        )
        num_quantize_units = sum(
            log.get("_num_quantize_units", zero) for log in logging_outputs
        )

        lm_score_sum = sum(log.get("_lm_score_sum", zero) for log in logging_outputs)
        vocab_seen = (
            sum(log.get("_vocab_seen", zero) for log in logging_outputs)
            .bool()
            .sum()
            .item()
        )
        kaldi_score_sum = sum(
            log.get("_kaldi_score_sum", zero) for log in logging_outputs
        )
        word_lm_sum = sum(log.get("_word_lm_sum", zero) for log in logging_outputs)

        metrics.log_scalar_sum("_num_char_errors", num_char_errors)
        metrics.log_scalar_sum("_num_chars", num_chars)
        metrics.log_scalar_sum("_num_word_errors", num_word_errors)
        metrics.log_scalar_sum("_num_words", num_words)

        metrics.log_scalar_sum("lm_score_sum", lm_score_sum)
        metrics.log_scalar_sum("num_pred_chars", num_pred_chars)

        metrics.log_scalar_sum("_num_precision_counts", num_precision_counts)
        metrics.log_scalar_sum("_num_recall_counts", num_recall_counts)
        metrics.log_scalar_sum("_num_segments", num_segments) 
        metrics.log_scalar_sum("_num_gt_precision_counts", num_gt_precision_counts)
        metrics.log_scalar_sum("_num_gt_recall_counts", num_gt_recall_counts)
        metrics.log_scalar_sum("_num_gt_segments", num_gt_segments)
        metrics.log_scalar_sum("_num_pred_segments", num_pred_segments)

        metrics.log_scalar_sum("_num_quantize_errors", num_quantize_errors)
        metrics.log_scalar_sum("_num_quantize_units", num_quantize_units)

        if self.cfg.word_kenlm_path is not None:
            metrics.log_scalar_sum("kaldi_score_sum", kaldi_score_sum)
            metrics.log_scalar_sum("word_lm_sum", word_lm_sum)

        if num_chars > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )

            if lm_score_sum < 0 and vocab_seen > 0:
                metrics.log_scalar("vocab_seen_pct", vocab_seen / self.num_symbols)

                metrics.log_derived(
                    "weighted_lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["lm_score_sum"].sum
                        / (
                            meters["num_pred_chars"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    )
                    / meters["vocab_seen_pct"].avg ** self.cfg.vocab_usage_power,
                )

                metrics.log_derived(
                    "lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["lm_score_sum"].sum
                        / (
                            meters["num_pred_chars"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    ),
                )
            else:
                metrics.log_derived("weighted_lm_ppl", lambda meters: float("inf"))

        if num_words > 0:
            if word_lm_sum != 0:
                metrics.log_derived(
                    "word_lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["word_lm_sum"].sum
                        / (
                            meters["_num_words"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    ),
                )
                metrics.log_derived(
                    "weighted_word_lm_ppl",
                    lambda meters: math.pow(
                        10,
                        -meters["word_lm_sum"].sum
                        / (
                            meters["_num_words"].sum + meters["nsentences"].sum
                        ),  # account for </s>
                    )
                    / meters["vocab_seen_pct"].avg ** self.cfg.vocab_usage_power,
                )

            if self.cfg.word_kenlm_path is not None:
                metrics.log_derived(
                    "kaldi_score",
                    lambda meters: meters["kaldi_score_sum"].sum
                    / meters["nsentences"].sum,
                )
       
        if num_segments > 0:
            metrics.log_derived(
                "boundary_precision",
                lambda meters: meters["_num_precision_counts"].sum
                * 100.0
                / meters["_num_pred_segments"].sum
                if meters["_num_pred_segments"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "boundary_recall",
                lambda meters: meters["_num_recall_counts"].sum
                * 100.0
                / meters["_num_segments"].sum
                if meters["_num_segments"].sum > 0
                else float("nan"),
            )

        if num_gt_segments > 0:
            metrics.log_derived(
                "gt_boundary_precision",
                lambda meters: meters["_num_gt_precision_counts"].sum
                * 100.0
                / meters["_num_pred_segments"].sum
                if meters["_num_pred_segments"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "gt_boundary_recall",
                lambda meters: meters["_num_gt_recall_counts"].sum
                * 100.0
                / meters["_num_gt_segments"].sum
                if meters["_num_gt_segments"].sum > 0
                else float("nan"),
            )

        if num_quantize_units > 0:
            metrics.log_derived(
                "quantize_uer",
                lambda meters: meters["_num_quantize_errors"].sum
                * 100.0
                / meters["_num_quantize_units"].sum
            )

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg)

        return model
