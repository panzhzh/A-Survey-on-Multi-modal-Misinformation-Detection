#!/usr/bin/env python3
# config.py
# -*- coding: utf-8 -*-

import os

# ============ 公共超参数 ============
EPOCHS = 3       # 如果 EPOCHS>0 则固定训练轮数，否则启用早停
PATIENCE = 3     # 早停时的最大不提升轮数

# ============ 基础路径 ============
BASE_DIR = "E:/pythonCode/MR2_baseline"

# 数据集 JSON 所在目录
queries_root = os.path.join(BASE_DIR, "queries_dataset_merge")
dataset_train_json = os.path.join(queries_root, "dataset_items_train.json")
dataset_val_json   = os.path.join(queries_root, "dataset_items_val.json")
dataset_test_json  = os.path.join(queries_root, "dataset_items_test.json")

# 特征文件保存目录
saved_npy_folder = os.path.join(BASE_DIR, "saved_npy_files")
os.makedirs(saved_npy_folder, exist_ok=True)

features_train = os.path.join(saved_npy_folder, "image_features_clip_train.npy")
features_val = os.path.join(saved_npy_folder, "image_features_clip_val.npy")
features_test = os.path.join(saved_npy_folder, "image_features_clip_test.npy")

# 其他：模型 / 输出路径
model_root = os.path.join(BASE_DIR, "methods", "clip")
best_dir   = os.path.join(model_root, "best_model")
output_dir = os.path.join(model_root, "output")