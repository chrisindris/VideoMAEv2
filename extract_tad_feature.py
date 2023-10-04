"""Extract features for temporal action detection datasets"""
import argparse
import os
import random

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms

import re
from pathlib import Path
import json

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


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
        '--save_path',
        default='/root/models/VideoMAEv2/thumos14_video/th14_vit_g_16_4',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='vit_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='/data/i5O/pretrained/VideoMAEv2/vit_g_hybrid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')
    # TODO: make the switching system between ActionFormer subset and full UCF101-THUMOS neater
    parser.add_argument(
        '--use_actionformer_subset',
        action='store_true'
    )
    parser.add_argument(
        '--actionformer_subset',
        default='/data/i5O/UCF101-THUMOS/THUMOS14/thumos/annotations/thumos14.json',
        help='this flag finds the 413 videos for ActionFormer'
    )

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    else:
        raise NotImplementedError()


def get_all_videos_in_subdirs(args):
    """ Find all of the videos (mp4, avi) below the data path.
    """

    # list of all files in all subdirs 
    all_files = []
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            all_files.append(os.path.join(root, file)) 

    # only get videos 
    reg = re.compile(".*\.mp4|.*\.avi")
    all_videos = list(filter(reg.search, all_files))

    return all_videos


def get_actionformer_subset(args):
    """ Return a list of the 413 videos used for actionformer.
    """
    with open(args.actionformer_subset, 'r') as f:
        data = json.load(f)

    return list(data['database'].keys()) 


def extract_feature(args):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()
    start_idx_range = get_start_idx_range(args.data_set)
    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])

    # get video path
    # vid_list = os.listdir(args.data_path)
    vid_list = get_all_videos_in_subdirs(args)
    
    random.shuffle(vid_list)

    # get model & load ckpt
    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(args.ckpt_path, map_location='cuda') # cpu
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    # extract feature
    
    actionformer_subset = get_actionformer_subset(args)
    num_videos = len(actionformer_subset)
    counter = 0

    for idx, vid_name in enumerate(vid_list):
        #url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        url = os.path.join(args.save_path, Path(vid_name).stem + '.npy')
        if os.path.exists(url) or (Path(vid_name).stem not in actionformer_subset):
            continue

        print(url)
        print(vid_name)
        video_path = vid_name # os.path.join(args.data_path, vid_name)
        vr = video_loader(video_path)
        counter += 1

        feature_list = []
        print(len(vr)) # number of frames ie 6234
        print(len(list(start_idx_range(len(vr))))) # number of frames being used ie 1555
        print(vr[0].shape) # size of first frame ie (180, 320, 3)=(height, width, color channels)
        print(list(start_idx_range(len(vr)))[0:10]) # list of frames to use ie [0,4,8,12,16,...]
        for start_idx in start_idx_range(len(vr)):
            data = vr.get_batch(np.arange(start_idx, start_idx + 16)).asnumpy()
            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
            input_data = frame_q.unsqueeze(0).cuda()

            with torch.no_grad():
                feature = model.forward_features(input_data)
                feature_list.append(feature.cpu().numpy()) # feature.cpu().numpy()

        # [N, C]
        np.save(url, np.vstack(feature_list))
        print(f'[{counter} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
