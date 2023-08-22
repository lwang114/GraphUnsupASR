import argparse
from praatio import textgrid
from praatio import audio


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--label_path")
    parser.add_argument("--split")
    parser.add_argument("--out_path")
    return parser


