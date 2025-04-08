#!/usr/bin/env python3
# model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


##############################################################################
# 通用的 HF 文本编码器
##############################################################################
class HFTextEncoder(nn.Module):
    """
    使用 HuggingFace Transformers 加载任意预训练模型（Encoder-Only）并取句向量。
    如果要兼容 GPT/T5/BART/LLaMA 等，需要额外做分支逻辑。
    """
    def __init__(self, pretrained_name="bert-base-uncased", max_length=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.model = AutoModel.from_pretrained(pretrained_name)
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

    def forward(self, list_of_texts):
        """
        Args:
          list_of_texts: list[str], B 条文本
        Returns:
          text_emb: [B, hidden_size], 取第0个token的向量作为整体句向量
        """
        inputs = self.tokenizer(
            list_of_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 这里简单地取第 0 个 token 的向量（通常是 [CLS] 对于 BERT/RoBERTa/ALBERT 等）
        cls_emb = last_hidden[:, 0, :]
        return cls_emb


##############################################################################
# 自定义损失函数（FocalLoss / BalancedCrossEntropyLoss），与原先一致
##############################################################################
class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            class_weights = [1.0, 1.0, 1.0]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, logits, labels):
        """
        logits: [B, num_classes]
        labels: [B]
        """
        cw = self.class_weights.to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        gather_vals = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        w = cw[labels]
        loss = - torch.sum(w * gather_vals) / labels.shape[0]
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, labels):
        """
        logits: [B, num_classes]
        labels: [B]
        """
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            ce_loss = F.cross_entropy(logits, labels, weight=alpha, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, labels, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


##############################################################################
# 文本分类网络
##############################################################################
class TextClassificationNetwork(nn.Module):
    """
    用 HFTextEncoder 提取文本表示，然后接一个分类头输出 num_classes。
    """
    def __init__(self, pretrained_name="bert-base-uncased", num_classes=3):
        super().__init__()
        # 初始化一个通用的 HFTextEncoder
        self.text_encoder = HFTextEncoder(
            pretrained_name=pretrained_name,
            max_length=512
        )
        self.hidden_size = self.text_encoder.hidden_size
        self.num_classes = num_classes

        # 简单全连接分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, list_of_texts):
        # 先编码文本
        emb = self.text_encoder(list_of_texts)  # [B, hidden_size]
        # 再过分类头
        logits = self.classifier(emb)           # [B, num_classes]
        return logits
