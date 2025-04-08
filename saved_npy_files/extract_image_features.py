#!/usr/bin/env python3
# extract_image_features.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# 从 transformers 中引入图像模型相关
from transformers import (
    AutoImageProcessor, AutoModel,
    Blip2Processor, Blip2Model,
    CLIPVisionModel, CLIPProcessor
)

# 你的自定义 Config
class Config:
    BASE_DIR = "E:/pythonCode/MR2_baseline"
    queries_root = os.path.join(BASE_DIR, "queries_dataset_merge")
    saved_npy_folder = os.path.join(BASE_DIR, "saved_npy_files")
    os.makedirs(saved_npy_folder, exist_ok=True)

    dataset_train_json = os.path.join(queries_root, "dataset_items_train.json")
    dataset_val_json   = os.path.join(queries_root, "dataset_items_val.json")
    dataset_test_json  = os.path.join(queries_root, "dataset_items_test.json")


ALL_MODELS = [
    "resnet101",
    "efficientnet",
    "densenet",
    "clip",
    "vit",
    "swin",
    "blip2"
]

IMAGE_MODEL_DICT = {
    "resnet101":    "microsoft/resnet-101",
    "efficientnet": "google/efficientnet-b7",
    "densenet":     "timm/densenet121.ra_in1k",
    "clip":         "openai/clip-vit-large-patch14",
    "vit":          "google/vit-base-patch16-224",
    "swin":         "microsoft/swin-base-patch4-window7-224",
    "blip2":        "Salesforce/blip2-flan-t5-xl"
}

def build_feature_path(model_name, split_type):
    file_name = f"image_features_{model_name}_{split_type}.npy"
    return os.path.join(Config.saved_npy_folder, file_name)

with open(Config.dataset_train_json, 'r', encoding='utf-8') as f:
    data_items_train = json.load(f)
with open(Config.dataset_val_json, 'r', encoding='utf-8') as f:
    data_items_val = json.load(f)
with open(Config.dataset_test_json, 'r', encoding='utf-8') as f:
    data_items_test = json.load(f)

def unify_path(path: str) -> str:
    path = os.path.normpath(path)
    path = path.replace("\\", "/")
    return path

def collect_main_image_paths(data_items_dict, root_dir):
    all_paths = set()
    for _, item in data_items_dict.items():
        rel_path = item.get('image_path', "")
        if not rel_path:
            continue
        abs_path = os.path.join(root_dir, rel_path)
        if os.path.isfile(abs_path):
            abs_path = unify_path(abs_path)
            all_paths.add(abs_path)
    return list(all_paths)

# ---------------------------
# 这里是核心改动：我们直接在 forward 里用官方 Processor
# ---------------------------
class ImageEncoder(nn.Module):
    def __init__(self, model_name:str):
        super().__init__()
        if model_name not in IMAGE_MODEL_DICT:
            raise ValueError(f"模型 {model_name} 不在映射表 IMAGE_MODEL_DICT 中！")
        self.model_name = model_name
        repo_id = IMAGE_MODEL_DICT[model_name]

        if model_name == "blip2":
            self.processor = Blip2Processor.from_pretrained(repo_id)
            self.base_model = Blip2Model.from_pretrained(repo_id)

        elif model_name == "clip":
            self.processor = CLIPProcessor.from_pretrained(repo_id)
            self.base_model = CLIPVisionModel.from_pretrained(repo_id)

        else:
            # 对于其他 (resnet101, efficientnet, densenet, vit, swin)
            self.processor = AutoImageProcessor.from_pretrained(repo_id)
            self.base_model = AutoModel.from_pretrained(repo_id)

        # 如果是 CNN，就用自适应池化
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, pil_images):
        """
        Args:
          pil_images: List of PIL Images (长度B)
        Returns:
          feats: Tensor[B, D]
        """
        # 1) 用官方 Processor 得到 pixel_values
        inputs = self.processor(images=pil_images, return_tensors="pt")

        # 注意：有的 Processor 可能返回 pixel_values 之外的 keys，如 images,或者 input_ids
        # 这里我们只取 pixel_values
        pixel_values = inputs["pixel_values"].to(self.base_model.device)

        # 2) 前向
        if self.model_name == "blip2":
            vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
            last_hidden = vision_outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            feats = last_hidden.mean(dim=1)

        elif self.model_name == "clip":
            outputs = self.base_model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            feats = hidden.mean(dim=1)

        else:
            outputs = self.base_model(pixel_values=pixel_values)
            hidden_state = outputs.last_hidden_state
            # hidden_state 可能是 [B, C, H, W] 或 [B, seq_len, hidden_dim]
            if hidden_state.ndim == 4:
                # CNN形式 (ResNet / EfficientNet / DenseNet等)
                pooled = self.pool(hidden_state)          # [B, C, 1, 1]
                feats = pooled[:, :, 0, 0]                # [B, C]
            elif hidden_state.ndim == 3:
                # Transformer形式 (ViT / Swin ...)
                feats = hidden_state.mean(dim=1)
            else:
                raise ValueError("模型输出维度不符合预期!")

        return feats


# 让 DataLoader 只返回 path，PIL Image；把处理放到 encoder.forward() 里
class ImgPathDataset(Dataset):
    def __init__(self, path_list):
        super().__init__()
        self.path_list = path_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert("RGB")
        return path, img

def collate_fn(batch):
    # batch: List[(path, pil_image), ...]
    paths = [b[0] for b in batch]
    imgs  = [b[1] for b in batch]
    return paths, imgs

def extract_and_save_features(path_list, image_encoder, save_path, batch_size=8, device='cpu'):
    if os.path.isfile(save_path):
        print(f"[跳过] {save_path} 已存在，无需重复提取。")
        return
    if len(path_list) == 0:
        print(f"[跳过] {save_path}，因为 path_list 为空。")
        return

    ds = ImgPathDataset(path_list)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=0)

    feature_dict = {}
    image_encoder.eval()
    image_encoder.to(device)

    with torch.no_grad():
        for paths, pil_imgs in tqdm(dl, desc=f"Extract {image_encoder.model_name}"):
            # 直接把整批 PIL 传给 image_encoder
            # 它会在内部调用 .processor(...)
            feats = image_encoder(pil_imgs)   # [B, D]
            feats = feats.cpu().numpy()
            for p, f in zip(paths, feats):
                feature_dict[p] = f

    np.save(save_path, feature_dict)
    print(f"[完成] {image_encoder.model_name} 的 {len(feature_dict)} 条图像特征 -> {save_path}")

def main():
    train_main_paths = collect_main_image_paths(data_items_train, Config.queries_root)
    val_main_paths   = collect_main_image_paths(data_items_val,   Config.queries_root)
    test_main_paths  = collect_main_image_paths(data_items_test,  Config.queries_root)

    print(f"Train 主图数: {len(train_main_paths)}")
    print(f"Val   主图数: {len(val_main_paths)}")
    print(f"Test  主图数: {len(test_main_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in ALL_MODELS:
        print("============================================")
        print(f"开始处理模型: {model_name}")
        print("============================================")

        encoder = ImageEncoder(model_name)

        train_save_path = build_feature_path(model_name, "train")
        val_save_path   = build_feature_path(model_name, "val")
        test_save_path  = build_feature_path(model_name, "test")

        extract_and_save_features(
            path_list=train_main_paths,
            image_encoder=encoder,
            save_path=train_save_path,
            batch_size=8,
            device=device
        )
        extract_and_save_features(
            path_list=val_main_paths,
            image_encoder=encoder,
            save_path=val_save_path,
            batch_size=8,
            device=device
        )
        extract_and_save_features(
            path_list=test_main_paths,
            image_encoder=encoder,
            save_path=test_save_path,
            batch_size=8,
            device=device
        )

if __name__ == "__main__":
    main()
