""" Provides some functions to work with the extracted .npy features.
"""

import argparse
import os
from pathlib import Path

from extract_tad_feature import get_actionformer_subset

def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='THUMOS14',
        choices=['THUMOS14', 'FINEACTION'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default='/data/i5O/UCF101-THUMOS/',
        type=str,
        help='dataset path')

    parser.add_argument(
        '--features_path',
        default='/root/models/VideoMAEv2/thumos14_video/th14_vit_g_16_4/',
        type=str, 
        help='path to the .npy features we are working with'
    )

    parser.add_argument(
        '--actionformer_subset',
        default='/data/i5O/UCF101-THUMOS/THUMOS14/thumos/annotations/thumos14.json',
        help='this flag finds the 413 functions[args.operation]()videos for ActionFormer'
    )

    parser.add_argument(
        '--operation',
        default='size_of_subset',
        choices=['count_all_extracted_features', 'size_of_subset', 'move', 'count_extracted_features'],
        help='choose operation'
    )

    return parser.parse_args()


def size_of_subset(args):
    return len(get_actionformer_subset(args))


def count_all_extracted_features(args):
    return len(os.listdir(args.features_path))


def get_extracted_features(args):
    all_files = [Path(feature).stem for feature in os.listdir(args.features_path)]
    thumos14_files = get_actionformer_subset(args) 
    return list(set(all_files).intersection(thumos14_files))

def count_extracted_features(args):
    return len(get_extracted_features(args))

def move(args):
    #TODO:
    print("WIP")


if __name__ == '__main__':
    args = get_args()
    print(globals()[args.operation](args))
