#!/usr/bin/env python3
# train_infer.py
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import config
import text_loader
import image_loader
from model import MultiModalNetwork, FocalLoss  # 示例：多模态模型 + 自定义loss
from monitor import TrainingMonitor

##############################################################################
# 1) Dataset
##############################################################################
class RumorDataset(Dataset):
    """
    同时加载文本+图像特征：对每条数据 (从 data_items_dict 里)：
      - 解析文本 (text_loader.build_text)
      - 读取图像特征 (image_loader.get_image_feature)
      - label
    """
    def __init__(self, data_items_dict, queries_root, image_feat_dict):
        super().__init__()
        self.data_items_dict = data_items_dict
        # 按 key 的数值大小排序
        self.keys = sorted(data_items_dict.keys(), key=int)
        self.queries_root = queries_root
        self.image_feat_dict = image_feat_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_items_dict[key]

        # 标签
        label = item.get("label", -1)
        label = torch.tensor(int(label), dtype=torch.long)

        # 读取 inverse_annotation / direct_annotation
        inv_path_abs = os.path.join(self.queries_root, item["inv_path"])
        inv_data = text_loader.load_inv_json(inv_path_abs)

        direct_path_abs = os.path.join(self.queries_root, item["direct_path"])
        direct_data = text_loader.load_direct_json(direct_path_abs)

        # 拼接文本
        caption = item.get("caption", "")
        text = text_loader.build_text(caption, inv_data, direct_data)

        # 图像特征
        rel_img_path = item.get("image_path", "")
        img_feat = image_loader.get_image_feature(
            self.image_feat_dict,
            self.queries_root,
            rel_img_path
        )

        sample = {
            "label": label,
            "text": text,
            "img_feat": img_feat,
            "unique_key": key
        }
        return sample


##############################################################################
# 2) Collate Function
##############################################################################
def collate_fn(batch):
    labels = []
    texts = []
    img_feats = []
    bkeys = []
    for s in batch:
        labels.append(s["label"])
        texts.append(s["text"])
        img_feats.append(s["img_feat"])
        bkeys.append(s["unique_key"])

    labels = torch.stack(labels, dim=0)        # [B]
    img_feats = torch.stack(img_feats, dim=0)  # [B, 1024] (根据具体特征维度)
    return labels, texts, img_feats, bkeys


##############################################################################
# 3) build_dataloader
##############################################################################
def build_dataloader(data_items_dict, image_feat_dict, split='train', batch_size=16):
    ds = RumorDataset(
        data_items_dict=data_items_dict,
        queries_root=config.queries_root,
        image_feat_dict=image_feat_dict
    )
    shuffle_flag = (split == 'train')  # 训练集需要 shuffle，验证/测试不需要
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag, collate_fn=collate_fn)


##############################################################################
# 4) 评估函数
##############################################################################
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """
    在给定 data_loader 上做评估，返回 loss/acc/f1。
    训练过程中的验证阶段会用到此函数。
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    for batch in data_loader:
        labels, texts, img_feats, bkeys = batch
        labels = labels.to(device)
        img_feats = img_feats.to(device)

        logits = model(texts, img_feats)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    # 如果确保只有3类，可以指定 labels=[0,1,2]；如果类别数可变，可通过 set(all_labels+all_preds) 来动态获取
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    model.train()
    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro
    }


##############################################################################
# 5) 训练过程
##############################################################################
def train_model(model, train_dl, val_dl, epochs, monitor=None, model_tag="model"):
    print(f"----- Start Training: {model_tag} -----")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 使用 FocalLoss 作为示例
    criterion = FocalLoss(gamma=2.0, alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1.2e-4)

    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        correct = 0
        total_ex = 0

        for step, batch in enumerate(train_dl, start=1):
            labels, texts, img_feats, bkeys = batch
            labels = labels.to(device)
            img_feats = img_feats.to(device)
            bs = labels.size(0)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
                logits = model(texts, img_feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == labels).sum().item()

            running_loss += loss.item() * bs
            correct += batch_correct
            total_ex += bs
            global_step += 1

            if monitor is not None:
                monitor.log_metrics({
                    "train_loss": loss.item(),
                    "train_acc": batch_correct/bs
                }, step=global_step)
                monitor.log_lr(optimizer, step=global_step)
                monitor.update_sps(bs, step=global_step)

        # 验证
        val_metrics = evaluate(model, val_dl, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc  = val_metrics["acc"]
        val_f1   = val_metrics["f1_macro"]

        if monitor is not None:
            monitor.log_metrics({
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1
            }, step=global_step)

        print(f"Epoch {epoch}/{epochs} - val_loss={val_loss:.4f} - val_acc={val_acc:.4f} - val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0

            best_ckpt = os.path.join(config.best_dir, f"{model_tag}_best.pt")
            os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve_count += 1

        # 如果 EPOCHS<=0，可以改成早停逻辑
        if config.EPOCHS <= 0:
            if no_improve_count >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training finished. best_f1={best_f1:.4f}, best_acc={best_acc:.4f} @ epoch {best_epoch}")


##############################################################################
# 6) 测试推理（仅评估最优模型，存结果到唯一 CSV）
##############################################################################
@torch.no_grad()
def evaluate_best_and_save(model, test_dl, device, model_tag, monitor=None):
    """
    加载 {model_tag}_best.pt，进行测试推理，并把结果（包括混淆矩阵等）保存到
    config.output_dir/best_results.csv。如果新F1高于已有记录，则更新，否则保持原样。
    """
    # 1) 加载最优模型
    best_ckpt = os.path.join(config.best_dir, f"{model_tag}_best.pt")
    if not os.path.isfile(best_ckpt):
        print(f"[Warning] best checkpoint not found: {best_ckpt}")
        return

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"[Test Eval] Loaded model from {best_ckpt}")

    model.eval()
    all_preds = []
    all_labels = []

    # 2) 前向推理
    for batch in test_dl:
        labels, texts, img_feats, bkeys = batch
        labels = labels.to(device)
        img_feats = img_feats.to(device)

        logits = model(texts, img_feats)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())

    # ========== FIX: 使用 labels=[0,1,2]，保证 cm 一定是 3×3 ==========
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    acc  = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)
    prec_macro = precision_score(all_labels, all_preds, average='macro',
                                 zero_division=0, labels=[0,1,2])  # FIX
    rec_macro  = recall_score(all_labels, all_preds, average='macro',
                              zero_division=0, labels=[0,1,2])    # FIX
    f1_macro   = f1_score(all_labels, all_preds, average='macro',
                          zero_division=0, labels=[0,1,2])        # FIX

    # 4) 获取 GPU usage & sps（如有 monitor）
    if monitor is not None:
        avg_gpu_util = monitor.get_avg_gpu_util()
        avg_gpu_mem  = monitor.get_avg_gpu_mem()
        avg_sps      = monitor.get_avg_sps()
    else:
        avg_gpu_util = 0.0
        avg_gpu_mem  = 0.0
        avg_sps      = 0.0

    # 5) 更新 best_results.csv（只存最优 F1 的那次）
    csv_path = os.path.join(config.output_dir, "best_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 读取已有的 CSV（若存在）
    if os.path.isfile(csv_path):
        df_old = pd.read_csv(csv_path)
    else:
        df_old = pd.DataFrame(columns=[
            "model_name", "accuracy", "precision", "recall", "f1_macro",
            "cm00", "cm01", "cm02", "cm10", "cm11", "cm12", "cm20", "cm21", "cm22",
            "gpu_usage", "gpu_mem", "sps"
        ])

    # 查找是否已存在本模型的记录
    row_idx = df_old.index[df_old["model_name"] == model_tag].tolist()
    if len(row_idx) > 0:
        old_f1 = df_old.loc[row_idx[0], "f1_macro"]
        if f1_macro > old_f1:
            df_old = _update_or_insert_model_row(df_old, model_tag, acc, prec_macro, rec_macro,
                                                 f1_macro, cm, avg_gpu_util, avg_gpu_mem,
                                                 avg_sps, idx=row_idx[0])
    else:
        df_old = _update_or_insert_model_row(df_old, model_tag, acc, prec_macro, rec_macro,
                                             f1_macro, cm, avg_gpu_util, avg_gpu_mem,
                                             avg_sps, idx=None)

    # 保存回 CSV
    df_old.to_csv(csv_path, index=False)
    print(f"Results updated to {csv_path} (only if new F1 is better).")


def _update_or_insert_model_row(df, model_name, acc, prec, rec, f1, cm,
                                gpu_usage, gpu_mem, sps, idx=None):
    """
    把新的评估结果更新到指定行或插入新行。cm 是 3x3 的混淆矩阵。
    """
    row_data = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "cm00": cm[0, 0], "cm01": cm[0, 1], "cm02": cm[0, 2],
        "cm10": cm[1, 0], "cm11": cm[1, 1], "cm12": cm[1, 2],
        "cm20": cm[2, 0], "cm21": cm[2, 1], "cm22": cm[2, 2],
        "gpu_usage": gpu_usage,
        "gpu_mem": gpu_mem,
        "sps": sps
    }
    if idx is not None:
        # 更新已有行
        for k, v in row_data.items():
            df.loc[idx, k] = v
    else:
        # 新增一行
        df.loc[len(df)] = row_data
    return df


##############################################################################
# 7) 主入口 + 子集函数
##############################################################################
def subset_data(data_dict, ratio=0.01):
    """
    取 dictionary 的前 ratio 比例，确保至少拿到 1 条。
    假设 data_dict 的 key 是整数字符串，按数值升序处理。
    """
    if ratio <= 0 or ratio >= 1:
        return data_dict
    keys_sorted = sorted(data_dict.keys(), key=int)
    limit = max(1, int(len(keys_sorted) * ratio))
    selected_keys = keys_sorted[:limit]
    new_data_dict = {k: data_dict[k] for k in selected_keys}
    return new_data_dict


def main(
    run_mode=2,
    text_model="bert-base-multilingual-cased",
    image_model="resnet101",
    fusion_level="early",
    use_small=False
):
    """
    Args:
      run_mode: 0=只训练, 1=只推理, 2=先训练再推理
      text_model:  "bert-base-multilingual-cased", "xlm-roberta-base", ...
      image_model: "resnet101","efficientnet","densenet","clip","vit","swin","blip2",...
      fusion_level: "early"/"mid"/"late"
      use_small: 是否仅用 1% 数据做快速测试
    """
    monitor = TrainingMonitor(
        project_name="MR2_baseline",
        run_name=f"{text_model}_{image_model}_{fusion_level}",
        gpu_index=0,
        enable_gpu_monitor=True,
        wandb_config={
            "run_mode": run_mode,
            "text_model": text_model,
            "image_model": image_model,
            "fusion_level": fusion_level,
            "epochs": config.EPOCHS,
            "use_small": use_small
        }
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 加载数据
    with open(config.dataset_train_json, 'r', encoding='utf-8') as f:
        data_items_train = json.load(f)
    with open(config.dataset_val_json, 'r', encoding='utf-8') as f:
        data_items_val = json.load(f)
    with open(config.dataset_test_json, 'r', encoding='utf-8') as f:
        data_items_test = json.load(f)

    # 如果只想做小规模测试，截取前 1%
    if use_small:
        data_items_train = subset_data(data_items_train, 0.1)
        data_items_val   = subset_data(data_items_val,   0.1)
        data_items_test  = subset_data(data_items_test,  0.1)

    # 2) 分别加载 train/val/test 的图像特征
    train_feat_path = os.path.join(config.saved_npy_folder, f"image_features_{image_model}_train.npy")
    val_feat_path   = os.path.join(config.saved_npy_folder, f"image_features_{image_model}_val.npy")
    test_feat_path  = os.path.join(config.saved_npy_folder, f"image_features_{image_model}_test.npy")

    if not os.path.isfile(train_feat_path):
        raise FileNotFoundError(f"train_feat_path not found: {train_feat_path}")
    if not os.path.isfile(val_feat_path):
        raise FileNotFoundError(f"val_feat_path not found: {val_feat_path}")
    if not os.path.isfile(test_feat_path):
        raise FileNotFoundError(f"test_feat_path not found: {test_feat_path}")

    train_feat_dict = np.load(train_feat_path, allow_pickle=True).item()
    val_feat_dict   = np.load(val_feat_path,   allow_pickle=True).item()
    test_feat_dict  = np.load(test_feat_path,  allow_pickle=True).item()

    # 3) 构建 DataLoader
    train_dl = build_dataloader(data_items_train, train_feat_dict, split='train', batch_size=16)
    val_dl   = build_dataloader(data_items_val,   val_feat_dict,   split='val',   batch_size=16)
    test_dl  = build_dataloader(data_items_test,  test_feat_dict,  split='test',  batch_size=16)

    # 4) 构建多模态模型
    model = MultiModalNetwork(
        text_model_name=text_model,
        image_model_name=image_model,
        fusion_level=fusion_level,
        num_classes=3
    )
    model_tag = f"{text_model}_{image_model}_{fusion_level}"

    # 5) 运行模式
    if run_mode == 0:
        # 只训练
        train_model(model, train_dl, val_dl, epochs=config.EPOCHS, monitor=monitor, model_tag=model_tag)
    elif run_mode == 1:
        # 只推理
        evaluate_best_and_save(model, test_dl, device, model_tag=model_tag, monitor=monitor)
    else:
        # 先训练，再推理
        train_model(model, train_dl, val_dl, epochs=config.EPOCHS, monitor=monitor, model_tag=model_tag)
        evaluate_best_and_save(model, test_dl, device, model_tag=model_tag, monitor=monitor)

    monitor.stop()


# 如果只想快速测试某个单一组合，可以在此直接调用 main()
if __name__ == "__main__":
    text_models = [
        "bert-base-multilingual-cased",
        "xlm-roberta-large",
        "distilbert-base-uncased"
    ]
    image_models = [
        "resnet101",
        "efficientnet",
        "densenet",
        "clip",
        "vit",
        "swin",
        "blip2"
    ]
    fusion_levels = ["early","mid", "late"]

    # 下述大循环总计 2×7×3=42 个组合，如果只想单测一个组合，请自行注释或修改
    for tm in text_models:
        for im in image_models:
            for fl in fusion_levels:
                main(
                    run_mode=2,
                    text_model=tm,
                    image_model=im,
                    fusion_level=fl,
                    use_small=False  # 仅用1%数据快速测试
                )
