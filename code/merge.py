import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from ipdb import set_trace as st
import os
import argparse


def merge_files(file1, file2):
    df1 = pd.read_csv(file1, delimiter="\t")
    df2 = pd.read_csv(file2, delimiter="\t")
    dd = pd.concat([df1, df2])
    dd = dd.sort_values(by=['idx'])
    merged_file = f'{file1}_merged'
    # st()
    dd.to_csv(merged_file, sep='\t', index=False)
    print(f'Successfully output: {merged_file}')

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--file1")
parser.add_argument("--file2")
args = parser.parse_args()

print(f'Merging {args}')
merge_files(args.file1, args.file2)
print()



