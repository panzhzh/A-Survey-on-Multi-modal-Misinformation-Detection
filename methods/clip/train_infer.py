#!/usr/bin/env python3
# train_infer.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import config
import text_loader
import image_loader
from model import MultiModalNetwork, FocalLoss  # 我们保留示例用的 FocalLoss
from monitor import TrainingMonitor


##############################################################################
# 1) Dataset
##############################################################################
class RumorDataset(Dataset):
    """
    同时加载文本+图像特征：对每条数据 (从 data_items_dict 里)：
      - 解析文本
      - 读取图像特征
      - label
    """
    def __init__(self, data_items_dict, queries_root, image_feat_dict):
        super().__init__()
        self.data_items_dict = data_items_dict
        self.keys = sorted(data_items_dict.keys(), key=int)
        self.queries_root = queries_root
        self.image_feat_dict = image_feat_dict

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_items_dict[key]

        label = item.get("label", -1)
        label = torch.tensor(int(label), dtype=torch.long)

        # 拼接文本（使用 text_loader 内的方法）
        inv_path_abs = os.path.join(self.queries_root, item["inv_path"])
        inv_data = text_loader.load_inv_json(inv_path_abs)

        direct_path_abs = os.path.join(self.queries_root, item["direct_path"])
        direct_data = text_loader.load_direct_json(direct_path_abs)

        caption = item.get("caption", "")
        text = text_loader.build_text(caption, inv_data, direct_data)

        # 读取图像特征（离线的 CLIP 特征）
        rel_img_path = item.get("image_path", "")
        img_feat = image_loader.get_image_feature(
            self.image_feat_dict,
            self.queries_root,
            rel_img_path
        )

        return {
            "label": label,
            "text": text,
            "img_feat": img_feat,
            "unique_key": key
        }


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
    img_feats = torch.stack(img_feats, dim=0)  # [B, D] or [B, N, D]
    return labels, texts, img_feats, bkeys


##############################################################################
# 3) DataLoader builder
##############################################################################
def build_dataloader(data_items_dict, image_feat_dict, split='train', batch_size=16):
    ds = RumorDataset(
        data_items_dict=data_items_dict,
        queries_root=config.queries_root,
        image_feat_dict=image_feat_dict
    )
    shuffle_flag = (split == 'train')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag, collate_fn=collate_fn)


##############################################################################
# 4) 验证函数
##############################################################################
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
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

    # 这里以 FocalLoss 为示例
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
                    "train_acc": batch_correct / bs
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

        # 保存最佳
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

        # 如果 EPOCHS<=0，可改成早停逻辑，这里保持原逻辑
        if config.EPOCHS <= 0:
            if no_improve_count >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training finished. best_f1={best_f1:.4f}, best_acc={best_acc:.4f} @ epoch {best_epoch}")


##############################################################################
# 6) 测试推理（评估最优模型 + 存结果）
##############################################################################
@torch.no_grad()
def evaluate_best_and_save(model, test_dl, device, model_tag, monitor=None):
    best_ckpt = os.path.join(config.best_dir, f"{model_tag}_best.pt")
    if not os.path.isfile(best_ckpt):
        print(f"[Warning] best checkpoint not found: {best_ckpt}")
        return

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"[Test Eval] Loaded model from {best_ckpt}")

    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_dl:
        labels, texts, img_feats, bkeys = batch
        labels = labels.to(device)
        img_feats = img_feats.to(device)

        logits = model(texts, img_feats)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    acc  = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)
    prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0, labels=[0,1,2])
    rec_macro  = recall_score(all_labels, all_preds, average='macro', zero_division=0, labels=[0,1,2])
    f1_macro   = f1_score(all_labels, all_preds, average='macro', zero_division=0, labels=[0,1,2])

    if monitor is not None:
        avg_gpu_util = monitor.get_avg_gpu_util()
        avg_gpu_mem  = monitor.get_avg_gpu_mem()
        avg_sps      = monitor.get_avg_sps()
    else:
        avg_gpu_util = 0.0
        avg_gpu_mem  = 0.0
        avg_sps      = 0.0

    # 更新 best_results.csv（若新F1更好则替换）
    csv_path = os.path.join(config.output_dir, "best_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.isfile(csv_path):
        df_old = pd.read_csv(csv_path)
    else:
        df_old = pd.DataFrame(columns=[
            "model_name", "accuracy", "precision", "recall", "f1_macro",
            "cm00", "cm01", "cm02", "cm10", "cm11", "cm12",
            "cm20", "cm21", "cm22",
            "gpu_usage", "gpu_mem", "sps"
        ])

    row_idx = df_old.index[df_old["model_name"] == model_tag].tolist()
    if len(row_idx) > 0:
        old_f1 = df_old.loc[row_idx[0], "f1_macro"]
        if f1_macro > old_f1:
            df_old = _update_or_insert_model_row(df_old, model_tag, acc, prec_macro,
                                                 rec_macro, f1_macro, cm,
                                                 avg_gpu_util, avg_gpu_mem, avg_sps,
                                                 idx=row_idx[0])
    else:
        df_old = _update_or_insert_model_row(df_old, model_tag, acc, prec_macro,
                                             rec_macro, f1_macro, cm,
                                             avg_gpu_util, avg_gpu_mem, avg_sps,
                                             idx=None)

    df_old.to_csv(csv_path, index=False)
    print(f"Results updated to {csv_path} (only if new F1 is better).")


def _update_or_insert_model_row(df, model_name, acc, prec, rec, f1, cm,
                                gpu_usage, gpu_mem, sps, idx=None):
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
        for k, v in row_data.items():
            df.loc[idx, k] = v
    else:
        df.loc[len(df)] = row_data
    return df


##############################################################################
# 7) main() 入口
##############################################################################
def subset_data(data_dict, ratio=0.01):
    """
    取 dictionary 的前 ratio 比例，至少 1 条
    """
    if ratio <= 0 or ratio >= 1:
        return data_dict
    keys_sorted = sorted(data_dict.keys(), key=int)
    limit = max(1, int(len(keys_sorted) * ratio))
    selected_keys = keys_sorted[:limit]
    return {k: data_dict[k] for k in selected_keys}


def main(
    run_mode=2,
    fusion_level="early",
    use_small=False
):
    """
    run_mode: 0=只训练, 1=只推理, 2=先训练再推理
    fusion_level: 'early' / 'mid' / 'late'
    use_small: 是否使用较小数据进行快速测试
    """
    monitor = TrainingMonitor(
        project_name="MR2_baseline",
        run_name=f"clip_{fusion_level}",
        gpu_index=0,
        enable_gpu_monitor=True,
        wandb_config={
            "run_mode": run_mode,
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

    if use_small:
        data_items_train = subset_data(data_items_train, 0.01)
        data_items_val   = subset_data(data_items_val,   0.01)
        data_items_test  = subset_data(data_items_test,  0.01)

    # 2) 加载离线的 CLIP 图像特征
    train_feat_dict = np.load(config.features_train, allow_pickle=True).item()
    val_feat_dict   = np.load(config.features_val,   allow_pickle=True).item()
    test_feat_dict  = np.load(config.features_test,  allow_pickle=True).item()

    # 3) DataLoader
    train_dl = build_dataloader(data_items_train, train_feat_dict, split='train', batch_size=16)
    val_dl   = build_dataloader(data_items_val,   val_feat_dict,   split='val',   batch_size=16)
    test_dl  = build_dataloader(data_items_test,  test_feat_dict,  split='test',  batch_size=16)

    # 4) 构建多模态模型（仅使用 CLIP）
    model = MultiModalNetwork(
        fusion_level=fusion_level,
        num_classes=3
    )
    model_tag = f"clip_{fusion_level}"

    # 5) 根据 run_mode 执行
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

if __name__ == "__main__":
    # 示例：在此直接调用 main()，可根据需要修改 run_mode / fusion_level 等
    main(run_mode=2, fusion_level="late", use_small=False)
