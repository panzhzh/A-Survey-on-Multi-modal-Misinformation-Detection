#!/usr/bin/env python3
# image_loader.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

def get_image_feature(feature_dict: dict, queries_root: str, rel_img_path: str):
    """
    根据相对路径 rel_img_path，在 queries_root 下组装绝对路径，然后
    到 feature_dict 中取对应的特征向量。找不到就返回长度为 0 的向量。
    """
    abs_path = os.path.join(queries_root, rel_img_path)
    abs_path = os.path.normpath(abs_path)
    abs_path = abs_path.replace("\\", "/")

    feat = feature_dict.get(abs_path, None)
    if feat is None:
        print(f"[Warning] No feature found for {abs_path}, using zero-len vector.")
        feat = np.zeros((0,), dtype='float32')

    output = f"{abs_path} {feat.sum()}\n"
    # 写入文件，使用追加模式
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(output)
    return torch.tensor(feat, dtype=torch.float32)
