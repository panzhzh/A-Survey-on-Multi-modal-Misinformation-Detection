#!/usr/bin/env python3
# graph_loader.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

def load_graph_features(path: str) -> dict:
    """
    读取 graph 的特征文件 (npy)，返回 {key: embedding_ndarray} 的字典。
    若不存在，会抛出 FileNotFoundError。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Graph feature file not found: {path}")
    data = np.load(path, allow_pickle=True).item()
    return data


def get_graph_feature(graph_feat_dict: dict, claim_key: str) -> torch.Tensor:
    """
    从 graph_feat_dict 中获取对应 claim_key 的图向量。
    如果没有找到，则返回一个全 0 的 256 维向量。
    """
    feat = graph_feat_dict.get(claim_key, None)
    if feat is None:
        feat = np.zeros((256,), dtype='float32')
    return torch.tensor(feat, dtype=torch.float32)
