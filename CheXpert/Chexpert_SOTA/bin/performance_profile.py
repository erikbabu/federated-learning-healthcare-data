import os
import sys
import argparse
import json
from easydict import EasyDict as edict

from thop import profile, clever_format

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from model.global_pool import ExpPool, LinearPool, LogSumExpPool, GlobalPool

import torch
from model.classifier import Classifier

parser = argparse.ArgumentParser(description='Profile model FLOPS/MACS')
parser.add_argument('--cfg_path', required=True, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('--file_name', default='profile_results.txt', type=str, help="Name of file to save results to")


def count_exp_pool(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    # exponentiation happens twice in exp_pool forward layer
    total_exp = 2 * nfeatures 

    # summation of all elements happens twice and once for eps value
    total_add = 2 * (nfeatures - 1) + 1

    # summation of all elements happens twice
    total_sub = 2 * (nfeatures - 1)

    # other multiplication happens once
    total_mul = nfeatures

    # division happens once
    total_div = nfeatures

    total_ops = batch_size * (total_exp + total_add + total_sub + total_mul + total_div)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_lin_pool(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()
 
    total_add = 2 * (nfeatures - 1) + 1
    total_mul = nfeatures
    total_div = nfeatures

    total_ops = batch_size * (total_add + total_mul + total_div)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


def count_log_sum_exp_pool(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = 2 * nfeatures 
    total_add = nfeatures - 1
    total_sub = nfeatures - 1
    total_mul = 3 * nfeatures + 1
    total_div = nfeatures + 1

    total_ops = batch_size * (total_exp + total_add + total_sub + total_mul + total_div)

    m.total_ops += torch.DoubleTensor([int(total_ops)])

def count_sig(m, x, y):
    # Using this approximation for exponetial operation:  exp(x) = 1 + x + x^2/2! + .. + x^9/9!
    # For sigmoid f(x) = 1/(1+exp(x)): there are totally 10 add ops, 9 division ops(2! are considered as constant).
    # Since it is element-wise operation. The final ops is about(10+9)*num_elements.
    x = x[0]

    nelements = x.numel()

    total_ops = 19 * nelements
    m.total_ops += torch.DoubleTensor([int(total_ops)])


if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
    
    model = Classifier(cfg)    
    
    dataloader_train = DataLoader(
                                    ImageDataset(cfg.train_csv, cfg, mode='train'),
                                    batch_size=cfg.train_batch_size, num_workers=4,
                                    drop_last=True, shuffle=False
                                 )

    device = torch.device("cpu")

    custom_ops = {
        ExpPool: count_exp_pool,
        LinearPool: count_lin_pool,
        LogSumExpPool: count_log_sum_exp_pool,
        torch.nn.modules.activation.Sigmoid: count_sig,
    }

    for data in dataloader_train:
            inputs = data[0].to(device)
            macs, params = profile(model, inputs=(inputs, ), custom_ops=custom_ops)
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