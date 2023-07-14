import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.svm import LinearSVC
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.io import read_video
from torchvision.transforms import (
    Normalize,
    Compose, 
    Resize,
    CenterCrop, 
    ToTensor,
)
from torchvision.transforms.functional import to_pil_image

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default="/home/hertin/data/MS-ASL",
    )
    parser.add_argument(
        "--metadata_path",
    )
    parser.add_argument(
        "--model_name", 
        default="vgg19",
    )
    parser.add_argument(
        "--out_path",
        default=""
    )
    parser.add_argument(
        "--classify", 
        action="store_true",
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
    )
    return parser

# Reference: https://pytorch.org/vision/0.9/models.html
class ResNet34FeatureReader(nn.Module):
    def __init__(self):
        super(ResNet34FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        resnet = models.resnet34(pretrained=True).to(self.device)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ]
        )
        
    def forward(self, imgs):
        x = []
        for img in imgs:
            img = self.transform(img)
            img = self.normalize(img)
            x.append(img)
        x = torch.stack(x).to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        out = out.squeeze(-1).squeeze(-1)
        return out   

# Reference: https://pytorch.org/vision/0.9/models.html
class ResNet152FeatureReader(nn.Module):
    def __init__(self):
        super(ResNet152FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        resnet152 = models.resnet152(pretrained=True).to(self.device)
        self.conv1 = resnet152.conv1
        self.bn1 = resnet152.bn1
        self.relu = resnet152.relu
        self.maxpool = resnet152.maxpool
        self.layer1 = resnet152.layer1
        self.layer2 = resnet152.layer2
        self.layer3 = resnet152.layer3
        self.layer4 = resnet152.layer4
        self.avgpool = resnet152.avgpool
        
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = Compose(
            [
                Resize(256), 
                CenterCrop(224), 
                ToTensor(),
            ]
        )
        
    def forward(self, imgs):
        x = []
        for img in imgs:
            img = self.transform(img)
            img = self.normalize(img)
            x.append(img)
        x = torch.stack(x).to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        out = out.squeeze(-1).squeeze(-1)
        return out

class VGG19FeatureReader(nn.Module):
    def __init__(self):
        super(VGG19FeatureReader, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Use {self.device}")
        vgg19 = models.vgg19(pretrained=True).to(self.device)
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        classifier = list(
            vgg19.classifier.children()
        )[:-2]
        self.extractor = nn.Sequential(*classifier)
        
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = Compose(
            [
                Resize(256), 
                CenterCrop(224), 
                ToTensor(),
            ]
        )
        
    def forward(self, imgs):
        x = []
        for img in imgs:
            img = self.transform(img)
            img = self.normalize(img)
            x.append(img)
        x = torch.stack(x).to(self.device)            
        #x = torch.zeros((1, 3, 224, 224), device=self.device)  # XXX
        x = self.features(x)
        x = self.avgpool(x)
        out = self.extractor(
            x.reshape(x.size(0), -1)
        )
        return out

# Extract features from image classifiers pretrained on ImageNet
# for the fingerspelling images and put it into the following 
# format:
#       root/
#           - {split}.tsv
#           - {split}.npy
def main():
    # Get input arguments
    parser = get_parser()
    args = parser.parse_args()
    data_path = Path(args.data_path)
    metadata_path = Path(args.metadata_path)
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    #(out_path / "train").mkdir(parents=True, exist_ok=True)
    #(out_path / "test").mkdir(parents=True, exist_ok=True)
    if args.model_name == "vgg19":
        model = VGG19FeatureReader()
    elif args.model_name == "resnet34":
        model = ResNet34FeatureReader()
    elif args.model_name == "resnet152":
        model = ResNet152FeatureReader()
    else:
        raise ValueError(f"model name {args.model_name} not found")
    words = None
    svm = LinearSVC()
    
    begin_time = time.time()
    for split in ["train"]:
        print(
            f"Extracting features for the {split} set",
            flush=True,
        )
        X = []
        y = []
        nf = 0
        # Extract features for the dataset
        if not (out_path / f"{split}.npy").exists():
            video_dict = json.load(open(metadata_path))
            vocab = sorted(video_dict)
            with open(out_path / f"{split}.tsv", "w") as f_tsv:
                print(str(data_path), file=f_tsv)
                for w, v_list in sorted(video_dict.items()):
                    for j, v in enumerate(v_list):
                        fname = v["url"].split("v=")[-1]
                        start = v["start_time"]
                        end = v["end_time"]
                        width, height = v["width"], v["height"]
                        y1, x1, y2, x2 = v["box"]
                        x1 = int(x1*width)
                        y1 = int(y1*height)
                        x2 = int(x2*width)
                        y2 = int(y2*height)
        
                        if args.debug and j > 1:
                            break                    
                        
                        fpath = data_path / f"{fname}.mp4"
                        if not fpath.exists():
                            fpath = data_path / f"{fname}.mkv"
                        print(fpath, flush=True)
                        try:
                            vid_frames = read_video(
                                str(fpath), 
                                start_pts=start,
                                end_pts=end,
                                pts_unit="sec",
                            )[0]
                            vid_frames = vid_frames[:, y1:y2, x1:x2] 
                            print(f"{fname}.mp4_{v['start_time']}_{v['end_time']}\t1", file=f_tsv)
                        except:
                            print(f"Warning: failed to read {fpath}")
                            continue
                                        
                        with torch.no_grad():
                            print(f"Number of frames: {vid_frames.size(0)}", flush=True)
                            vid_frames = [
                                Image.fromarray(frame.numpy())
                                for frame in vid_frames
                            ]
                            dur = len(vid_frames)
                            if dur % 100 == 0:
                                n_b = dur // 100
                            else:
                                n_b = dur // 100 + 1
                            x = []
                            for b in range(n_b):
                                x.append(
                                    model(vid_frames[b*100:(b+1)*100])
                                )
                            x = torch.cat(x)
                            if dur != x.size(0):
                                print(f"Warning: {fpath} output size {x.size(0)} != input duration {dur}")
                            x = x.mean(0)

                            X.append(
                                x.cpu().detach().numpy()
                            )
                            y.append(v["label"])
                        if (nf + 1) % 1000 == 0:
                            print(
                                f"Extract features for {nf} files"
                                f" in {time.time() - begin_time:.2f} s",
                                flush=True,
                            )
                        nf += 1
            
            # Save the feature
            X = np.stack(X)
            np.save(
                out_path / f"{split}.npy", X
            )
        else:
            X = np.load(out_path / f"{split}.npy")
            with open(out_path / f"{split}.tsv") as f_tsv:
                lines = f_tsv.read().strip().split("\n")
                _ = lines.pop(0)
                y_strs = [l.split("\t")[0].split("_")[-1] for l in lines]
                vocab = list(set(y_strs))
                y = [vocab.index(w) for w in y_strs]

        # Train a linear classifier using the features
        if args.classify:            
            y = np.asarray(y)
            if split == "train":
                print(
                    "Start training a linear classifier on the train set",
                    flush=True,
                )
                svm.fit(X, y)
            y_pred = svm.predict(X)
            print(
                f"{split} accuracy: {np.mean(y == y_pred):.4f}",
                flush=True,
            )

if __name__ == "__main__":
    main() 
