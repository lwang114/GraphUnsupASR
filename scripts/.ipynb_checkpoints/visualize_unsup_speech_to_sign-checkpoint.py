from librosa import display
from collections import defaultdict
from copy import deepcopy
import cv2
import json
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import pdb
import random
import torch
import torchvision
random.seed(1)
font = {'size': 15}
matplotlib.rc('font', **font)


class ASL_LibriSpeech_Visualizer:
    def __init__(
        self,
        retrieval_json,
        video_root,
        flow_root,
        speech_manifest_dir="/home/lwang114/workplace/fall2022/UnsupSpeech2Sign/manifest/asl_librispeech960_100words", 
        sent_manifest_dir="/home/lwang114/workplace/fall2022/UnsupSpeech2Sign/manifest/asl_librispeech960_100words/asl_feat/i3d_flow_charades_cpc_npredicts3_32negatives",
        word_manifest_dir="/home/lwang114/workplace/fall2022/UnsupSpeech2Sign/manifest/MS-ASL/i3d_flow_charades_mean_cpc",
        split="dev",
        out_root="/home/lwang114/workplace/fall2022/UnsupSpeech2Sign/visualizations",
    ):
        self.video_root = Path(video_root)
        self.flow_root = Path(flow_root)
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        speech_manifest_dir = Path(speech_manifest_dir)
        self.audio_data = []
        with open(speech_manifest_dir / f"{split}.tsv", "r") as f_tsv,\
            open(speech_manifest_dir / f"{split}.jsonlines", "r") as f_json:
            lines_tsv = f_tsv.read().strip().split("\n")
            _ = lines_tsv.pop(0)
            lines_json = f_json.read().strip().split("\n")
            for l_tsv, l_json in zip(lines_tsv, lines_json):
                audio_path = l_tsv.split("\t")[0]
                audio_dict = json.loads(l_json)
                self.audio_data.append(
                    (audio_path, audio_dict)
                )
    
        word_manifest_dir = Path(word_manifest_dir)
        sent_manifest_dir = Path(sent_manifest_dir)
        
        self.word_feats = np.load(word_manifest_dir / "train.npy")
        self.sent_feats = np.load(sent_manifest_dir / f"{split}.npy")

        self.word_data = []
        self.sent_data = []
        with open(word_manifest_dir / "train.tsv", "r") as f_wrd,\
            open(sent_manifest_dir / f"{split}.lengths", "r") as f_len:
            lines = f_wrd.read().strip().split("\n")
            _ = lines.pop(0)
            self.word_data = [l.split("\t")[0] for l in lines]
            sizes = list(map(int, f_len.read().strip().split("\n")))
            offset = 0
            for size in sizes:
                self.sent_data.append((offset, size))
                offset += size
    
        self.vid2info = {}
        with open(sent_manifest_dir / "../../wrd2vid.json", "r") as f_msasl:
            wrd2vid = json.load(f_msasl)
            i = 0
            for w in wrd2vid:
                for info in wrd2vid[w]:
                    video_id = info["url"].split("v=")[-1]
                    i += 1
                    start, end = info["start_time"], info["end_time"] 
                    video_id = f"{video_id}.mp4_{start}_{end}"
                    self.vid2info[video_id] = {"box": info["box"]}
    
    def show_similarity_map(
            self, idx, find_correct=True,
        ):
        align_indices, dist_mat, tran = sample_alignment(retrieval_file, wrd_file)
    
    def show_speech(self, idx):
        y, sr = read_speech(*self.audio_data[idx])
        plot_mel_spectrogram(y, sr)
        audio_path = Path(self.audio_data[idx][0])
        fname = audio_path.name.split(".")[0]
        plt.savefig(self.out_root / f"{fname}.png")
        plt.close()
    
    def show_video(self, idx):
        audio_path = Path(self.audio_data[idx][0])
        fname = audio_path.name.split(".")[0]
        offset, size = self.sent_data[idx]
        sent_ids = match_video_feature_with_id(
            self.sent_feats[offset:offset+size], 
            self.word_feats,
            self.word_data,
        )
        videos = read_video(self.video_root, sent_ids)
        boxes = [self.vid2info[video_id]["box"] for video_id in sent_ids]
        videos = extract_regions(videos, boxes)     
        video_cat = []
        for sent_id, v in zip(sent_ids, videos):
            for i, v_frame in enumerate(v[1::len(v) // 3]):
                v_frame = cv2.resize(v_frame, (224, 224)) 
                video_cat.append(v_frame)
        video_cat = np.concatenate(video_cat, axis=1)
        
        fig, ax = plt.subplots(figsize=(60, 5))
        plt.imshow(video_cat)
        plt.axis("off")
        plt.savefig(self.out_root / f"{fname}_rgb.png")
        plt.close()
    
    def show_optical_flow(self, idx):
        audio_path = Path(self.audio_data[idx][0])
        fname = audio_path.name.split(".")[0]
        offset, size = self.sent_data[idx]
        sent_ids = match_video_feature_with_id(
            self.sent_feats[offset:offset+size], 
            self.word_feats,
            self.word_data,
        )
        videos = read_video(self.video_root, sent_ids)
        boxes = [self.vid2info[video_id]["box"] for video_id in sent_ids]
        videos = extract_regions(videos, boxes)
        video_cat = []
        for sent_id, v in zip(sent_ids, videos):
            frame_idxs = np.arange(len(v))[1::len(v) // 3]
            print(frame_idxs)  # XXX
            flows = extract_optical_flows(v, frame_idxs)
            for i in range(flows.shape[1]):
                hsv = np.zeros((224, 224, 3))
                hsv[..., 1] = 255
                
                mag, ang = cv2.cartToPolar(
                    np.asarray(flows[0, i]), np.asarray(flows[1, i])
                )
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                hsv = hsv.astype(np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                rgb = np.zeros(bgr.shape)
                for c in range(3):
                    rgb[:, :, c] = bgr[:, :, 2-c] 
                video_cat.append(rgb)
                
        video_cat = np.concatenate(video_cat, axis=1)
        plt.figure(figsize=(60, 5))
        plt.imshow(video_cat)
        plt.axis("off")
        plt.savefig(self.out_root / f"{fname}_flow.png")
        np.save(self.out_root / f"{fname}_flow.npy", video_cat)
        plt.close()

def plot_similarity(S, tran):
    n_a, n_v = S.shape
    vmax = S.max()
    vmin = S.min()
    fig, ax = plt.subplots()
    S = S / S.sum(-1, keepdims=True)
    plt.pcolor(
      S, 
      vmin=vmin,
      vmax=vmax, 
      cmap=plt.cm.Blues, 
      edgecolors='k',
    )
    ax.set_xticks(np.arange(1, n_v+1)-0.5)
    ax.set_yticks(np.arange(1, n_a+1)-0.5)
    ax.set_xticklabels(tran, rotation=45)
    ax.set_yticklabels(tran, rotation=45)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.invert_yaxis()
    plt.colorbar()
        
def read_speech(audio_path, audio_dict, sr=16e3):
    """
    Args:
        audio_dict: a dict with fields
            utterance_id: str, name of the audio file
            words: a list of strs
            begins: a list of floats
            ends: a list of floats
    """
    utt_id = audio_dict["utterance_id"]
    begins = audio_dict["begins"]
    ends = audio_dict["ends"]
    raw_audio = librosa.load(
        audio_path, 
        sr=sr,
    )[0].squeeze()
    y = []
    for b, e in zip(begins, ends):
        y.append(raw_audio[int(b*sr):int(e*sr)])
    return y, sr

def plot_mel_spectrogram(y, sr=16e3):
    S = librosa.feature.melspectrogram(
        y=np.concatenate(y), sr=sr,
        win_length=400,
        hop_length=160,
        n_fft=512,
        n_mels=40,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 2))
    img = display.specshow(S_dB, x_axis='time',
        y_axis='mel', sr=sr,
        win_length=400,
        hop_length=160,
        n_fft=512,
        fmax=8000)
    
    offset = 0
    for seg in y[:-1]:
        offset += len(seg) / sr
        plt.axvline(x=offset, color="white")
    plt.axis("off")
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')

def match_video_feature_with_id(sent_feats, word_feats, word_ids):
    """
    Args:
        sent_feats: np.array of shape (n1, d)
        word_feats: np.array of shape (n2, d)
        word_ids: list of video ids for the word videos
    """    
    S = sent_feats @ word_feats.T
    S /= np.linalg.norm(word_feats, axis=-1)
    S /= np.linalg.norm(sent_feats, axis=-1)[:, np.newaxis]
    max_ind = S.argmax(-1)
    sent_ids = [word_ids[i] for i in max_ind]
    return sent_ids

def read_video(video_root, video_ids):
    """
    Args:
        video_root: str,
        video_ids: list of strs, each in the format
            {youtube id}.mp4_{start time}_{end time}
    """
    video_root = Path(video_root)
    videos = []
    for vid in video_ids:
        fname, time_stamps = vid.split(".mp4")
        start, end = list(map(float, time_stamps.split("_")[1:]))
        try:
            fpath = str(video_root / f"{fname}.mp4")
            video = torchvision.io.read_video(
                fpath, 
                start_pts=start,
                end_pts=end,
                pts_unit="sec",
            )[0]
        except:
            fpath = str(video_root / f"{fname}.m4v")
            video = torchvision.io.read_video(
                fpath,
                start_pts=start,
                end_pts=end,
                pts_unit="sec",
            )[0]
        videos.append(video.numpy())
    return videos

def extract_regions(videos, boxes):
    regions = []
    for v, box in zip(videos, boxes):
        x, y = v.shape[2], v.shape[1]
        y0 = int(y * box[0])
        x0 = int(x * box[1])
        y1 = int(y * box[2])
        x1 = int(x * box[3])
        regions.append(v[:, y0:y1, x0:x1])
    return regions

def extract_optical_flows(vid_frames, frame_idxs=None):
    flow = []
    
    if frame_idxs is None:
        frame_idxs = range(1, vid_frames.shape[0])
    
    for i in frame_idxs:
        i = max(1, i)
        prev = vid_frames[i-1]
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev = cv2.resize(prev,(224,224))
    
        curr = vid_frames[i]
        curr = cv2.resize(curr,(224,224))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    
        tmp_flow = compute_TVL1(prev, curr)
        tmp_flow = np.swapaxes(tmp_flow[None], 0, -1)[:,:,:,0]
        flow.append(tmp_flow)
        # prev = curr

    flow = np.asarray(flow)
    flow = np.swapaxes(flow, 0, 1)
    flow = (flow/255.)*2 - 1

    return torch.from_numpy(flow.astype("float32"))

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -20,20) #default values are +20 and -20

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

if __name__ == "__main__":
    retrieval_file = "/home/lwang114/workplace/fall2022/UnsupSpeech2Sign/multirun/l1_w2vu_word_100_segmented_onehot_clus400_asl_cpc_clus100_i3d_flow_charades/0/retrieval.json" 
    video_root = "/home/hertin/data/MS-ASL/downloads"
    flow_root = "/home/lwang114/workplace/fall2022/UnsupSpeech2Sign/manifest/MS-ASL/i3d_flow_charades"

    visualizer = ASL_LibriSpeech_Visualizer(
        retrieval_file, video_root, flow_root,
    )
    
    idx = 0
    print("Show mel spectrogram")
#    visualizer.show_speech(idx)
    print("Show video")
    visualizer.show_video(idx)
    print("Show optical flow")
#    visualizer.show_optical_flow(idx)