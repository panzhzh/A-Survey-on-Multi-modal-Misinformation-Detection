#!/usr/bin/env python3
# check_features.py
# -*- coding: utf-8 -*-

import os
import numpy as np

def print_first_10_items(npy_path):
    """读取 npy 并打印前 10 条 path 与特征和."""
    if not os.path.isfile(npy_path):
        print(f"[警告] 文件不存在: {npy_path}")
        return

    # 读取 npy (里面是一个 dict: {abs_path -> 特征向量})
    data_dict = np.load(npy_path, allow_pickle=True).item()
    print(f"\n===== 读取 {npy_path}，共 {len(data_dict)} 条 =====")

    count = 0
    for abs_path, feat in data_dict.items():
        print(abs_path, feat.sum())
        count += 1
        if count >= 10:
            break

def main():
    # 这里根据实际文件名拼接
    train_npy = "E:\pythonCode\MR2_baseline\saved_npy_files\image_features_resnet101_test.npy"
    val_npy   = "image_features_resnet101_val.npy"
    test_npy  = "image_features_resnet101_test.npy"

    # 依次打印
    print_first_10_items(train_npy)
    print_first_10_items(val_npy)
    print_first_10_items(test_npy)

if __name__ == "__main__":
    main()
