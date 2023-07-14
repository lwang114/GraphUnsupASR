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


def show_optical_flow(video_path, out_prefix, box=None):
    video_id = Path(video_path).name.split(".")[0]
    try:
        fpath = str(video_path)
        v = torchvision.io.read_video(
            fpath,
            pts_unit="sec",
        )[0].numpy()
    except:
        fpath = str(video_path)
        v = torchvision.io.read_video(
            fpath,
            pts_unit="sec",
        )[0].numpy()
    v = extract_box(v, box)
    frame_idxs = np.arange(len(v))[3::len(v) // 4]
    v_ds = v[frame_idxs]
    t, w, h, c = v_ds.shape
    v_ds = v_ds.reshape(t*w, h, c)
    plt.figure(figsize=(5, 20))
    plt.imshow(v_ds)
    plt.axis("off")
    plt.savefig(out_prefix+"_rgb.png")
    
    flows = extract_optical_flows(v, frame_idxs)
    rgb_cat = []
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
        rgb_cat.append(rgb)
    rgb_cat = np.concatenate(rgb_cat, axis=0)
    plt.figure(figsize=(5, 20))
    plt.imshow(rgb_cat)
    plt.axis("off")
    plt.savefig(out_prefix+"_optical_flow.png")

def extract_optical_flows(vid_frames, frame_idxs=None):
    flow = []    
    if frame_idxs is None:
        frame_idxs = range(1, vid_frames.shape[0])
    
    for i in frame_idxs:
        print(i)  # XXX
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

def extract_box(v, box):
    if box is None:
        return v
    x, y = v.shape[2], v.shape[1]
    y0 = int(y*box[0])
    x0 = int(x*box[1])
    y1 = int(y*box[2])
    x1 = int(x*box[3])
    return v[:, y0:y1, x0:x1]

if __name__ == "__main__":
    show_optical_flow(
        "../computer-science.mp4",
        "../visualizations/computer-science",
        box = [0, 0.3, 1, 0.75],
    )