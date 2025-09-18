import os
import random
import logging
import torch
import numpy as np
import os.path as osp
import yaml
import os.path as osp

class DictToObject:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = DictToObject(v)
            setattr(self, k, v)

class Cfg:
    def __init__(self, file_name):
        file_path = osp.join(get_root_dir(), 'conf', file_name)
        with open(file_path, "r") as f:
            conf = yaml.safe_load(f)
            self.model_args = DictToObject(conf.get('model_args', {}))
            self.conv_args = DictToObject(conf.get('conv_args', {}))
            self.seq_transformer_args = DictToObject(conf.get('seq_transformer_args', {}))
            self.run_args = DictToObject(conf.get('run_args', {}))
            self.dataset_args = DictToObject(conf.get('dataset_args', {}))


def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("Next_POI_data_Preprocess")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname