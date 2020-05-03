import os
import sys
import argparse
import json
from easydict import EasyDict as edict

from thop import profile, clever_format

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from torch.utils.data import DataLoader
from data.dataset import ImageDataset

import torch
from model.classifier import Classifier

parser = argparse.ArgumentParser(description='Profile model FLOPS/MACS')
parser.add_argument('--cfg_path', required=True, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('--file_name', default='profile_results.txt', type=str, help="Name of file to save results to")


if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
    
    dataloader_train = DataLoader(
                                    ImageDataset(cfg.train_csv, cfg, mode='train'),
                                    batch_size=cfg.train_batch_size, num_workers=4,
                                    drop_last=True, shuffle=False
                                 )

    device = torch.device("cpu")
    model = Classifier(cfg)    
    for data in dataloader_train:
            inputs = data[0].to(device)
            macs, params = profile(model, inputs=(inputs, ))
            break

    steps = len(dataloader_train)
    epochs = cfg.epoch
    total_batches = steps * epochs
    
    # When comparing MACs /FLOPs, we want the number to be implementation-agnostic and as general as possible. 
    # The THOP library therefore only considers the number of multiplications and ignore all other operations.
    total_macs = macs * total_batches
    total_flops_approx =  2 * total_macs

    total_macs_formatted, _ = clever_format([total_macs, params], "%.5f")
    total_flops_approx_formatted, _ = clever_format([total_flops_approx, params], "%.5f")

    print(f"Total MACs: {total_macs_formatted}")
    print(f"Approximate Total FLOPs: {total_flops_approx_formatted}")

    # Save results to file
    with open(args.file_name, "w") as f:
        f.write(f"Total MACs: {total_macs_formatted}\n")
        f.write(f"Approximate Total FLOPs: {total_flops_approx_formatted}")