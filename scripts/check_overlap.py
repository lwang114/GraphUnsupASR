import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file1")
parser.add_argument("file2")
args = parser.parse_args()

with open(args.file1, "r") as f1,\
    open(args.file2, "r") as f2:
    uids1 = set(f1.read().strip().split("\n"))
    uids2 = set(f2.read().strip().split("\n"))

    print(f"Number of overlapped uids: {len(uids1.intersection(uids2))}")

