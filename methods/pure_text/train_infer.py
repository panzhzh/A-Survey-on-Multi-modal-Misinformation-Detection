#!/usr/bin/env python3
# train_infer.py
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import config
import text_loader
from model import TextClassificationNetwork, FocalLoss, BalancedCrossEntropyLoss

# 如果不需要 W&B，可以去掉
from monitor import TrainingMonitor


##############################################################################
# 1) Dataset
##############################################################################
class RumorDataset(Dataset):
    """
    从 data_items_dict 里取出每条样本信息，并调用 text_loader.build_text()
    得到单一的 'text'。
    """
    def __init__(self, data_items_dict, queries_root):
        super().__init__()
        self.data_items_dict = data_items_dict
        # 按 key 的数值大小排序
        self.keys = sorted(data_items_dict.keys(), key=int)
        self.queries_root = queries_root

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data_items_dict[key]

        # 标签
        label = item.get("label", -1)
        label = torch.tensor(int(label), dtype=torch.long)

        # inverse_annotation / direct_annotation 的路径
        inv_path_abs = os.path.join(self.queries_root, item["inv_path"])
        inv_data = text_loader.load_inv_json(inv_path_abs)

        direct_path_abs = os.path.join(self.queries_root, item["direct_path"])
        direct_data = text_loader.load_direct_json(direct_path_abs)

        # Caption + inverse + direct 组合到一起得到一个字段: text
        caption = item.get("caption", "")
        text = text_loader.build_text(caption, inv_data, direct_data)

        sample = {
            "label": label,
            "unique_key": key,
            "text": text
        }
        return sample


##############################################################################
# 2) Collate Function
##############################################################################
def collate_fn(batch):
    labels = []
    texts = []
    bkeys = []
    for s in batch:
        labels.append(s["label"])
        texts.append(s["text"])
        bkeys.append(s["unique_key"])

    labels = torch.stack(labels, dim=0)
    return labels, texts, bkeys


##############################################################################
# 3) build_dataloader
##############################################################################
def build_dataloader(data_items_dict, split='train', batch_size=16, is_test=False):
    ds = RumorDataset(data_items_dict, queries_root=config.queries_root)
    # 训练集需要 shuffle，验证/测试集不需要
    shuffle_flag = (split == 'train') and (not is_test)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag, collate_fn=collate_fn)


##############################################################################
# 4) 评估函数
##############################################################################
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_labels_cpu = []
    all_preds_cpu = []

    for batch in data_loader:
        labels, texts, bkeys = batch
        labels = labels.to(device)

        # 前向推理
        logits = model(texts)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

        all_labels_cpu.extend(labels.cpu().numpy().tolist())
        all_preds_cpu.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    overall_acc = total_correct / total_samples

    prec_macro = precision_score(all_labels_cpu, all_preds_cpu, average='macro', zero_division=0)
    rec_macro  = recall_score(all_labels_cpu, all_preds_cpu, average='macro', zero_division=0)
    f1_macro   = f1_score(all_labels_cpu, all_preds_cpu, average='macro', zero_division=0)

    metrics_dict = {
        'loss': avg_loss,
        'acc': overall_acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'preds': all_preds_cpu,
        'labels': all_labels_cpu
    }
    model.train()
    return metrics_dict


##############################################################################
# 5) 训练
##############################################################################
def do_train(model, train_dl, val_dl, epochs, monitor=None, model_name="bert-base-uncased"):
    print("----- Start Training -----")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # 这里用 FocalLoss，也可改成 BalancedCrossEntropyLoss 或官方 CrossEntropyLoss
    criterion = FocalLoss(gamma=2.0, alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1.2e-4)

    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 自动混合精度
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_acc = 0.0
    best_f1  = 0.0  # 用于判断是否更新最优模型
    best_epoch = 0
    no_improve_count = 0
    global_step = 0  # 记录训练的 step（批次数），W&B 的 step 也用它

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0

        # === 训练 ===
        for step, batch in enumerate(train_dl, start=1):
            labels, texts, bkeys = batch
            labels = labels.to(device)
            bs = labels.size(0)

            optimizer.zero_grad()
            # 混合精度上下文
            with torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
                logits = model(texts)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()

            epoch_loss += loss.item() * bs
            epoch_correct += correct
            epoch_samples += bs
            global_step += 1  # 每个batch step+1

            # 若使用 monitor (W&B 等)，可记录日志
            if monitor is not None:
                monitor.log_metrics({
                    "train_loss": loss.item(),
                    "train_acc": correct / bs
                }, step=global_step)
                monitor.log_lr(optimizer, step=global_step)
                monitor.update_sps(bs, step=global_step)

        # === 验证 ===
        val_metrics = evaluate(model, val_dl, criterion, device)
        val_loss = val_metrics['loss']
        val_acc  = val_metrics['acc']
        val_f1   = val_metrics['f1_macro']

        # 注意：这里也用 global_step，以避免 step 回退的问题
        if monitor is not None:
            monitor.log_metrics({
                "val_loss": val_loss,
                "val_acc":  val_acc,
                "val_f1_macro": val_f1
            }, step=global_step)

        # === 检查是否是新最优 ===
        if (val_f1 > best_f1) or (val_f1 == best_f1 and val_acc > best_acc):
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0

            # 保存最优模型 => {model_name}_best.pt
            best_ckpt = os.path.join(config.best_dir, f"{model_name}_best.pt")
            os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve_count += 1

        # 早停逻辑
        if config.EPOCHS <= 0:
            if no_improve_count >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print("----- Training Finished -----")
    print(f"Best val f1={best_f1:.5f}, acc={best_acc:.5f} at epoch={best_epoch}")


##############################################################################
# 6) 测试推理（仅评估最优模型，存结果到 CSV）
##############################################################################
@torch.no_grad()
def evaluate_best_and_save(model, test_dl, device, model_name, monitor=None):
    """
    加载 {model_name}_best.pt，进行测试推理，并把结果（包括混淆矩阵等）保存到 CSV。
    若出现新F1高于已有记录，则更新 CSV，否则保持原样。
    """
    # 1) 加载最优模型
    best_ckpt = os.path.join(config.best_dir, f"{model_name}_best.pt")
    if not os.path.isfile(best_ckpt):
        print(f"[Warning] best checkpoint not found at {best_ckpt}.")
        return

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"[Test Eval] Loaded model from {best_ckpt}")

    model.eval()
    all_preds = []
    all_labels = []

    # 2) 前向推理
    for batch in test_dl:
        labels, texts, bkeys = batch
        labels = labels.to(device)

        logits = model(texts)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())

    # 3) 计算指标 & 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    acc  = sum(1 for t, p in zip(all_labels, all_preds) if t == p) / len(all_labels)
    prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec_macro  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # 4) 获取 GPU usage & sps（如有 monitor）
    if monitor is not None:
        avg_gpu_util = monitor.get_avg_gpu_util()  # 平均 GPU 利用率
        avg_gpu_mem  = monitor.get_avg_gpu_mem()   # 平均显存占用
        avg_sps      = monitor.get_avg_sps()       # 平均 sps
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
    row_idx = df_old.index[df_old["model_name"] == model_name].tolist()
    if len(row_idx) > 0:
        old_f1 = df_old.loc[row_idx[0], "f1_macro"]
        if f1_macro > old_f1:
            df_old = _update_or_insert_model_row(df_old, model_name, acc, prec_macro, rec_macro,
                                                 f1_macro, cm, avg_gpu_util, avg_gpu_mem,
                                                 avg_sps, idx=row_idx[0])
    else:
        df_old = _update_or_insert_model_row(df_old, model_name, acc, prec_macro, rec_macro,
                                             f1_macro, cm, avg_gpu_util, avg_gpu_mem,
                                             avg_sps, idx=None)

    # 保存回 CSV
    df_old.to_csv(csv_path, index=False)
    print(f"Results updated to {csv_path} (if new F1 is better).")


def _update_or_insert_model_row(df, model_name, acc, prec, rec, f1, cm,
                                gpu_usage, gpu_mem, sps, idx=None):
    """
    把新的评估结果更新到指定行或插入新行。
    cm 是 3x3 的混淆矩阵。
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
        # 新增一行: 用 loc 来插入
        df.loc[len(df)] = row_data
    return df


##############################################################################
# 7) 主入口
##############################################################################
def main(run_mode=2, model_name="bert-base-uncased"):
    """
    run_mode:
      0 => 只训练
      1 => 只推理
      2 => 先训练再推理

    model_name:
      指定想使用的预训练模型，比如 "bert-base-uncased" / "roberta-large" / ...
    """
    # 初始化监控器（若不需要可删除）
    monitor = TrainingMonitor(
        project_name="MR2_baseline",
        run_name=f"{model_name}",
        gpu_index=0,
        enable_gpu_monitor=True,
        wandb_config={
            "run_mode": run_mode,
            "batch_size": 16,
            "epochs": config.EPOCHS,
            "model_name": model_name
        }
    )
    # 读数据
    with open(config.dataset_train_json, 'r', encoding='utf-8') as f:
        data_items_train = json.load(f)
    with open(config.dataset_val_json, 'r', encoding='utf-8') as f:
        data_items_val = json.load(f)
    with open(config.dataset_test_json, 'r', encoding='utf-8') as f:
        data_items_test = json.load(f)

    # dataloader
    train_dl = build_dataloader(data_items_train, 'train', batch_size=16, is_test=False)
    val_dl   = build_dataloader(data_items_val,   'val',   batch_size=16, is_test=False)
    test_dl  = build_dataloader(data_items_test,  'test',  batch_size=16, is_test=True)

    # 模型
    model = TextClassificationNetwork(
        pretrained_name=model_name,
        num_classes=3
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if run_mode == 0:
        # 只训练
        do_train(model, train_dl, val_dl, epochs=config.EPOCHS, monitor=monitor, model_name=model_name)

    elif run_mode == 1:
        # 只推理(加载已有的最优模型并评估)
        evaluate_best_and_save(model, test_dl, device, model_name, monitor=monitor)

    else:
        # 先训练，再推理
        do_train(model, train_dl, val_dl, epochs=config.EPOCHS, monitor=monitor, model_name=model_name)
        evaluate_best_and_save(model, test_dl, device, model_name, monitor=monitor)

    # 停止监控器
    monitor.stop()


if __name__ == "__main__":
    # 要遍历的模型列表
    model_list = [
        "bert-base-uncased",
        "bert-base-cased",
        "bert-base-multilingual-cased",
        "roberta-base",
        "roberta-large",
        "xlm-roberta-base",
        "xlm-roberta-large",
        "albert-base-v2",
        "albert-large-v2",
        "distilbert-base-uncased",
        "xlnet-base-cased"
    ]

    # 逐个模型执行 main(...), run_mode=2 => 先训练再推理
    for model_name in model_list:
        print(f"\n\n===== Now running model: {model_name} =====")
        main(run_mode=2, model_name=model_name)
        print(f"===== Finished model: {model_name} =====\n\n")
