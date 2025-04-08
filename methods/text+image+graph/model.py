#!/usr/bin/env python3
# model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

##############################################################################
# 1) 通用的 HF 文本编码器 (修改以便拿到全序列表示)
##############################################################################
class HFTextEncoder(nn.Module):
    """
    使用 HuggingFace Transformers 加载任意预训练模型（Encoder-Only）。
    可返回:
    - 整个序列隐藏向量: [B, L, hidden_size]
    - [CLS] 向量: [B, hidden_size]
    """
    def __init__(self, pretrained_name="bert-base-uncased", max_length=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.model = AutoModel.from_pretrained(pretrained_name)
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length

    def forward(self, list_of_texts, return_cls_only=False):
        """
        Args:
          list_of_texts: list[str], B 条文本
          return_cls_only: 是否只返回 [CLS] 向量 (形状 [B, hidden_size])

        Returns:
          如果 return_cls_only=True,  返回 [B, hidden_size]
          如果 return_cls_only=False, 返回 [B, L, hidden_size]
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            list_of_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden_size]

        if return_cls_only:
            # 取第 0 个 token (CLS) 的向量
            cls_emb = last_hidden[:, 0, :]
            return cls_emb
        else:
            # 返回整个序列表示
            return last_hidden


##############################################################################
# 2) 自定义损失（FocalLoss / BalancedCrossEntropyLoss）
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
# 3) CrossAttentionFusion (保留示例)
##############################################################################
class CrossAttentionFusion(nn.Module):
    """
    一个小型双向 Cross-Attention 模块，用于文本序列和图像序列的交互。
    如果后面想扩展到三模态，可以多写一套 cross-attn，但示例仅做双模态。
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn_t2i = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, 
                                                   dropout=dropout, batch_first=True)
        self.cross_attn_i2t = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, 
                                                   dropout=dropout, batch_first=True)
        
        self.ln_text = nn.LayerNorm(d_model)
        self.ln_img = nn.LayerNorm(d_model)

    def forward(self, text_emb, img_emb, text_mask=None, img_mask=None):
        """
        Args:
            text_emb: [B, L, d]
            img_emb:  [B, I, d]
            text_mask, img_mask: 若有需要可传入 (batch_first=True 时是 [B, L] / [B, I])
        Return:
            text_out: [B, L, d]
            img_out:  [B, I, d]
        """
        # 1) 以文本为 Query，看图像 Key/Value
        t2i_out, _ = self.cross_attn_t2i(
            query=text_emb,
            key=img_emb,
            value=img_emb,
            key_padding_mask=img_mask  # 可选
        )
        text_out = self.ln_text(text_emb + t2i_out)

        # 2) 以图像为 Query，看文本 Key/Value
        i2t_out, _ = self.cross_attn_i2t(
            query=img_emb,
            key=text_emb,
            value=text_emb,
            key_padding_mask=text_mask  # 可选
        )
        img_out = self.ln_img(img_emb + i2t_out)

        return text_out, img_out


##############################################################################
# 4) MultiModalNetwork：修改以支持三模态 (文本+图像+graph)
##############################################################################
class MultiModalNetwork(nn.Module):
    def __init__(
        self,
        text_model_name="bert-base-uncased",
        image_model_name="resnet101",
        fusion_level="early",
        num_classes=3,
        num_heads=4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_level = fusion_level.lower()
   
        # ============ 文本编码器 ============
        self.text_encoder = HFTextEncoder(pretrained_name=text_model_name)
        self.text_hidden_size = self.text_encoder.hidden_size

        # ============ 图像投影层：动态创建 ============
        self.img_projector = None  # forward()中第一次见到img_feats时创建

        # ============ Graph 投影层：动态创建 ============
        # 假设 graph_emb 大小与 text_hidden_size 对齐
        self.graph_projector = None

        # ============ 不同融合方式的下游结构 ============
        if self.fusion_level == "early":
            # 三模态 early: 把 text/img/graph 三者全拼接 -> MLP
            # 最终维度 = text_hidden_size * 3
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size * 3),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size * 3, num_classes)
            )

        elif self.fusion_level == "mid":
            # 使用文本-图像的 Cross-Attention 做中期融合，
            # 然后再和 graph_emb 做简单拼接或加法
            self.cross_attn = CrossAttentionFusion(
                d_model=self.text_hidden_size,
                n_heads=num_heads
            )
            # Cross-Attention 之后取文本侧 [CLS]，
            # 然后再和 graph_emb 拼接(2*hidden_size)
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size * 2, num_classes)
            )

        elif self.fusion_level == "late":
            # 三模态 late: text_logits + image_logits + graph_logits (可学习的加权)
            # 分别建 3 个 head，然后加权平均
            self.text_classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size, num_classes)
            )
            self.image_classifier = None  # forward时第一次见到img_feats后建
            self.graph_classifier = None  # forward时第一次见到graph_feats后建

            # 3个可学习权重
            # 会在 forward() 用 softmax 归一化
            self.late_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)

        else:
            raise ValueError(f"Unknown fusion_level={fusion_level}")

    def forward(self, texts, img_feats, graph_feats=None):
        """
        Args:
          texts: list[str], B 条文本
          img_feats: Tensor[B, D_img] 或 [B, N, D_img]
          graph_feats: Tensor[B, D_g] 或 None
        Returns:
          logits: [B, num_classes]
        """
        device = next(self.parameters()).device

        # =====================================================================
        # 1) 文本表征
        # =====================================================================
        if self.fusion_level == "mid":
            # mid 融合时需要文本序列
            text_seq = self.text_encoder(texts, return_cls_only=False) 
        else:
            # early / late 只用 [CLS] 即可
            text_emb = self.text_encoder(texts, return_cls_only=True)  # [B, hidden_size]

        # =====================================================================
        # 2) 图像表征
        # =====================================================================
        if self.img_projector is None:
            # 第一次 forward 时根据 img_feats 的末尾维度创建投影层
            in_dim = img_feats.shape[-1]
            out_dim = self.text_hidden_size
            self.img_projector = nn.Linear(in_dim, out_dim).to(device)

        # 如果是 [B, N, D_img]，先做 mean pooling -> [B, D_img]
        if img_feats.dim() == 3:
            img_feats = img_feats.mean(dim=1)
        img_emb = self.img_projector(img_feats)  # [B, hidden_size]

        # =====================================================================
        # 3) Graph 表征 (可能为空)
        # =====================================================================
        if graph_feats is not None:
            if self.graph_projector is None:
                # 同样创建一个线性投影，保证输出维度 = text_hidden_size
                in_dim = graph_feats.shape[-1]
                out_dim = self.text_hidden_size
                self.graph_projector = nn.Linear(in_dim, out_dim).to(device)
            graph_emb = self.graph_projector(graph_feats)  # [B, hidden_size]
        else:
            # 如果没有graph_feats，就当它是个全0
            graph_emb = torch.zeros_like(img_emb)

        # =====================================================================
        # 4) 根据 fusion_level 不同，实现三模态融合
        # =====================================================================
        if self.fusion_level == "early":
            # -------------------- EARLY FUSION --------------------
            # text_emb + img_emb + graph_emb -> 拼接
            # text_emb: [B, hidden_size]
            # img_emb:  [B, hidden_size]
            # graph_emb:[B, hidden_size]
            # => [B, 3*hidden_size]
            fuse = torch.cat([text_emb, img_emb, graph_emb], dim=1)
            logits = self.classifier(fuse)

        elif self.fusion_level == "mid":
            # -------------------- MID FUSION --------------------
            # 1) 对文本序列和图像做 cross-attn
            #    text_seq: [B, L, hidden_size]
            #    img_emb: [B, hidden_size] => [B, 1, hidden_size]
            img_seq = img_emb.unsqueeze(1)
            fused_text_seq, fused_img_seq = self.cross_attn(text_seq, img_seq)

            # 2) 取文本侧 [CLS] + graph_emb 拼接
            fused_cls = fused_text_seq[:, 0]   # [B, hidden_size]
            fuse = torch.cat([fused_cls, graph_emb], dim=1)  # => [B, 2*hidden_size]
            logits = self.classifier(fuse)

        else:
            # -------------------- LATE FUSION --------------------
            # 1) 先分别得到 text_logits, image_logits, graph_logits
            # 如果没有graph_feats，这里 graph_emb=0，对应分类器还会给出某种输出
            if self.image_classifier is None:
                self.image_classifier = nn.Sequential(
                    nn.LayerNorm(self.text_hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.text_hidden_size, self.num_classes)
                ).to(device)

            if self.graph_classifier is None:
                self.graph_classifier = nn.Sequential(
                    nn.LayerNorm(self.text_hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.text_hidden_size, self.num_classes)
                ).to(device)

            text_logits  = self.text_classifier(text_emb)   # [B, num_classes]
            image_logits = self.image_classifier(img_emb)   # [B, num_classes]
            graph_logits = self.graph_classifier(graph_emb) # [B, num_classes]

            # 2) late_weights 经过 softmax -> alpha,beta,gamma
            alpha_beta_gamma = torch.softmax(self.late_weights, dim=0)  # shape=[3]
            # 三模态加权求和
            logits = (alpha_beta_gamma[0] * text_logits
                     +alpha_beta_gamma[1] * image_logits
                     +alpha_beta_gamma[2] * graph_logits)

        return logits
