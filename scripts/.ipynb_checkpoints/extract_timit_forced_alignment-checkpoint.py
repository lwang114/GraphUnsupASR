import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path")
    parser.add_argument("--out_path")

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    

