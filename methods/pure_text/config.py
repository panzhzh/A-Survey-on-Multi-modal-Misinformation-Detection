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

# 特征文件保存目录（如果仅做文本，可不需要 npy 特征，也可保留文件夹）
saved_npy_folder = os.path.join(BASE_DIR, "saved_npy_files")
os.makedirs(saved_npy_folder, exist_ok=True)

# 其他：模型 / 输出路径
model_root = os.path.join(BASE_DIR, "methods", "pure_text")
best_dir   = os.path.join(model_root, "best_model")
output_dir = os.path.join(model_root, "output")