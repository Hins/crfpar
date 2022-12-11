# -*- coding: utf-8 -*-

from parser.cmds import Predict
from parser.config import Config

import os
import torch
import argparse

def main(input, result):
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    args = parser.parse_args()
    args.conf = "config.ini"
    args.preprocess = False
    args.seed = 1
    args.threads = 16
    args.partial = True
    args.tree = True
    args.feat =  "tag"
    args.buckets = 3

    torch.set_num_threads(16)
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    args.mode = "predict"
    args.input = input
    args.result = result
    args.fields = os.path.join('./model/', 'fields')
    args.marg = True
    args.proj = True
    args.prob = True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Override the default configs with parsed arguments")
    args = Config("config.ini").update(vars(args))
    print(args)

    print(f"Run the subcommand in mode predict")
    cmd = Predict()
    return cmd(args)
