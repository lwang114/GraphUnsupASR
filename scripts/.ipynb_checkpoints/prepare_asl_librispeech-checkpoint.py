import argparse
from collections import defaultdict
import json
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path", 
        help="directory containing metadata of the raw speech dataset",
    )
    parser.add_argument(
        "--align_path",
        help="directory containing word alignments of the speech dataset",
    )
    parser.add_argument(
        "--video_path",
        help="directory containing word-level sign videos",
    )
    parser.add_argument("--split")
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument(
        "--out_path",
        help="directory containing the output metadata files", 
    )
    
    return parser


def read_alignment(ali_path):
    alignments = {}
    with open(ali_path, "r") as f_ali:
        for line in f_ali:
            utt_id, raw_text, raw_ali = line.split()
            text = raw_text.lower().strip("\"").split(",")
            ali = [float(t) for t in raw_ali.strip("\"").split(",")]
            alignments[utt_id] = {
                "utterance_id": utt_id, 
                "words": text,
                "begins": [0.0]+ali[:-1],
                "ends": ali,
            }
    return alignments


def filter_alignment(ali, vocab):
    filtered = {
        "utterance_id": ali["utterance_id"],
        "words": [],
        "begins": [],
        "ends": [],
    }

    for w, b, e in zip(ali["words"], ali["begins"], ali["ends"]):
        if w in vocab:
            filtered["words"].append(w)
            filtered["begins"].append(b)
            filtered["ends"].append(e)
    return filtered


def main():
    """
    Create metadata files for concatenated ASL word datasets:
        * wrd2vid.json: stores a dict of {word}: list of videos for the word 
        * {split}.tsv: contains path to the filtered spoken sentences
        * {split}.trn: contains word-level transcripts with only words from the
            vocabulary of the video dataset
        * {split}.jsonline: contains dicts of word-level forced alignment for each
            filtered spoken sentence   

    Args:
        manifest_path: str
        align_path: str
        video_path: str
        split: str
        out_path: str
    """
    parser = get_parser()
    args = parser.parse_args()
    manifest_path = Path(args.manifest_path)
    align_path = Path(args.align_path)
    video_path = Path(args.video_path)
    out_path = Path(args.out_path)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    print("Creating a mapping from words to sign videos ...")
    wrd2vid = defaultdict(list)
    n_vids = 0
    if not (out_path / "wrd2vid.json").exists():
        with open(out_path / "wrd2vid.json", "w") as f_wrd2vid:
            for x in ["train", "val", "test"]:
                with open(video_path / f"MSASL_{x}.json", "r") as f_vid:
                    vids = json.load(f_vid)
                    for vid in vids:
                        label = vid["label"]
                        if args.topk > 0 and label >= args.topk:
                            continue
                        wrd = vid["text"]
                        vid_name = vid["url"].split("v=")[-1]+".mp4"
                        if (video_path / "downloads" / vid_name).exists(): 
                            wrd2vid[wrd].append(vid)
                            n_vids += 1
            json.dump(wrd2vid, f_wrd2vid, indent=2, sort_keys=True)
    else:
        wrd2vid = json.load(open(out_path / "wrd2vid.json"))
        n_vids = sum(len(v) for v in wrd2vid.values())
    print(f"Number of videos: {n_vids}")
    print(f"Vocab size: : {len(wrd2vid)}")

    print("Creating a concatenated word dataset from LibriSpeech ...")
    alignments = {}
    n_utts = 0
    n_words = 0
    vocab = set()
    with open(out_path / f"{args.split}.tsv", "w") as f_out_tsv,\
        open(out_path / f"{args.split}.jsonlines", "w") as f_out_ali,\
        open(out_path / f"{args.split}.trn", "w") as f_out_wrd:
        f_out_tsv.write("./\n")
        for fn in sorted(manifest_path.iterdir()):
            audio_split = fn.name.split(".")[0]
            if args.split in audio_split and "tsv" in fn.name:
                with open(manifest_path / fn, "r") as f_tsv:
                    lines = f_tsv.read().strip().split("\n")
                    root = lines.pop(0)
                    for line in lines:
                        audio_path = Path(line.split("\t")[0])
                        utt_id = audio_path.name.split(".")[0]
                        audio_dir = Path(audio_path.parent)
                        if not utt_id in alignments: 
                            for ali_fn in (align_path / audio_split / audio_dir).iterdir():
                                if "alignment.txt" in ali_fn.name:
                                    alignments = read_alignment(align_path / audio_split / ali_fn)
                                    break
                            if not utt_id in alignments:
                                print(f"Failed to find alignment for {utt_id}")
                                continue
                        
                        alignment = filter_alignment(alignments[utt_id], wrd2vid)
                        if len(alignment["words"]) > args.min_length:
                            f_out_tsv.write(f"{root}/{line}\n")
                            f_out_ali.write(json.dumps(alignment)+"\n")
                            f_out_wrd.write(" ".join(alignment["words"])+"\n")
                            vocab.update(set(alignment["words"]))
                            n_words += len(alignment["words"]) 
                            n_utts += 1
    print(f"Vocab. size: {len(vocab)}")
    print(f"Number of filtered words for {args.split}: {n_words}")
    print(f"Number of filtered utterances for {args.split}: {n_utts}")


if __name__ == "__main__":
    main()
