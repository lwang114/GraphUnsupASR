import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.svm import LinearSVC
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import (
    Normalize,
    Compose, 
    Resize,
    CenterCrop, 
    ToTensor,
)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default="/home/lwang114/data/asl_alphabet",
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
        x = self.features(x)
        x = self.avgpool(x)
        out = self.extractor(
            x.view(x.size(0), -1)
        )
        return out

# Cluster the single-hand sign images using feature embeddings 
# pretrained on ImageNet
def main():
    # Get input arguments
    parser = get_parser()
    args = parser.parse_args()
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "train").mkdir(parents=True, exist_ok=True)
    (out_path / "test").mkdir(parents=True, exist_ok=True)
    model = VGG19FeatureReader()
    letters = None
    svm = LinearSVC()
    
    begin_time = time.time() 
    for split in ["train", "test"]:
        print(f"Extracting features for the {split} set")
        split_path = data_path / f"asl_alphabet_{split}"
        if split == "train":
            letters = [p.name for p in sorted(split_path.iterdir())]
        X = []
        y = []
        nf = 0
        # Extract features for the dataset
        for i, ltr in enumerate(letters):
            if split == "train":
                ltr_path = split_path / ltr
                filenames = ltr_path.iterdir()
            else:
                filenames = [split_path / f"{ltr}_test.jpg"]
            for j, fn in enumerate(filenames):
                if args.debug and (j > 1 or i > 5):
                    break
                img = Image.open(ltr_path / fn)
                x = model([img]).squeeze(0)
                # Save the feature
                np.save(
                    out_path / split / f"{fn.stem}.npy", 
                    x.detach().numpy(),
                )
                X.append(x)
                y.append(i)
                if (nf + 1) % 100 == 0:
                    print(
                        f"Extract features for {nf} files"
                        f" in {time.time() - begin_time:.2f} s"
                    )
                nf += 1
        
        # Train a linear classifier using the feature
        if args.classify:
            X = torch.stack(X).detach().numpy()
            y = np.asarray(y)
            if split == "train":
                print("Start training a linear classifier on the train set")
                svm.fit(X, y)
            y_pred = svm.predict(X)
            print(f"{split} accuracy: {np.mean(y == y_pred):.4f}")

if __name__ == "__main__":
    main() 
