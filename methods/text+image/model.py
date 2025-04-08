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
# 2) 自定义损失（FocalLoss / BalancedCrossEntropyLoss）保留即可
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
# 3) GatedFusion (原有示例仍可保留, 若不再需要可删)
##############################################################################
class GatedFusion(nn.Module):
    """
    对中期融合做一个示例：在文本向量和图像向量拼接前，先用两个 gating 简单融合。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_t = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gate_i = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, text_emb, img_emb):
        gate_t = torch.sigmoid(self.gate_t(text_emb))  # [B, hidden_dim]
        gate_i = torch.sigmoid(self.gate_i(img_emb))   # [B, hidden_dim]
        fused = gate_t * text_emb + gate_i * img_emb
        return fused


##############################################################################
# 4) 交叉注意力模块 (用于更成熟的中期融合)
##############################################################################
class CrossAttentionFusion(nn.Module):
    """
    一个小型双向 Cross-Attention 模块，用于文本序列和图像序列的交互。
    假设两侧维度均为 [B, T, hidden] 和 [B, I, hidden]。
    如果图像只有一个全局特征，可视为 I=1。
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
            text_mask, img_mask: 若有需要可传入 (batch_first=True 时通常是 [B, L] / [B, I])
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
# 5) 统一的多模态模型：三种融合方式（动态映射图像特征）
#    - early: 同之前不变
#    - mid: 采用 Cross-Attention
#    - late: 同之前不变
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
        """
        在 forward() 第一次拿到 img_feats 之后动态创建投影层。
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_level = fusion_level.lower()
   
        # 初始化文本编码器
        self.text_encoder = HFTextEncoder(pretrained_name=text_model_name)
        self.text_hidden_size = self.text_encoder.hidden_size

        # 动态建图像投影层
        self.img_projector = None  # 等 forward() 时第一次得到 img_feats 维度后再建

        # 根据融合方式创建相应模块
        if self.fusion_level == "early":
            # 直接拼接 -> MLP 分类
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size * 2, num_classes)
            )

        elif self.fusion_level == "mid":
            # 使用 Cross-Attention 做中期融合
            self.cross_attn = CrossAttentionFusion(d_model=self.text_hidden_size, n_heads=num_heads)

            # 注意：在 mid 模式中，我们需要对文本拿到全序列 (不再只用 cls)
            # 并对图像做投影后 -> [B, I, hidden], 这里 I=1 (一个全局图像向量)
            # 交叉注意力后，取文本侧 [CLS] 或池化作为最终表示。
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size, num_classes)
            )

        elif self.fusion_level == "late":
            # 文本 logits & 图像 logits -> 可学习 alpha 加权
            self.text_classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size, num_classes)
            )
            self.image_classifier = None  # 动态创建
            self.late_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        else:
            raise ValueError(f"Unknown fusion_level={fusion_level}")

    def forward(self, texts, img_feats):
        """
        Args:
          texts: list[str], B 条文本
          img_feats: Tensor[B, D_img] 或 [B, N, D_img]，
                     其中 D_img 不一定是 1024；N 为若干图像特征 (若只有1个特征则N=1)
        Returns:
          logits: [B, num_classes]
        """
        device = next(self.parameters()).device

        if self.fusion_level == "early":
            # 1) 文本只需 [CLS]
            text_emb = self.text_encoder(texts, return_cls_only=True)  # [B, hidden_size]
            # 2) 动态构造图像投影层
            if self.img_projector is None:
                in_dim = img_feats.shape[-1]  # 如果 [B, D_img] 或 [B, N, D_img]，取末尾维度
                out_dim = self.text_hidden_size
                self.img_projector = nn.Linear(in_dim, out_dim).to(device)

            # 若是 [B, D_img]，投影后得到 [B, hidden_size]
            # 若是 [B, N, D_img]，可以先做 mean pooling 或 flatten -> [B, D_img] 再投影
            # 这里演示最简单做法：如果是 [B, N, D_img] 就先 mean pooling 到 [B, D_img]
            if img_feats.dim() == 3:
                img_feats = img_feats.mean(dim=1)  # [B, D_img]
            img_emb = self.img_projector(img_feats)  # [B, hidden_size]

            # 3) 拼接 -> 分类
            fuse = torch.cat([text_emb, img_emb], dim=1)  # [B, 2*hidden_size]
            logits = self.classifier(fuse)

        elif self.fusion_level == "mid":
            # =============================
            # 1) 文本: 取全序列 [B, L, hidden]
            # =============================
            text_seq = self.text_encoder(texts, return_cls_only=False) 
            # shape: [B, L, hidden_size]

            # =============================
            # 2) 图像：动态投影 -> [B, I, hidden]
            # =============================
            # 如果原本是 [B, D_img]，则 I=1
            # 如果是 [B, N, D_img]，则 I=N
            if self.img_projector is None:
                in_dim = img_feats.shape[-1]
                out_dim = self.text_hidden_size
                self.img_projector = nn.Linear(in_dim, out_dim).to(device)

            # 若 [B, D_img], 直接投影到 [B, hidden_size], 再 unsqueeze 到 [B, 1, hidden_size]
            # 若 [B, N, D_img], 则投影到 [B, N, hidden_size]
            if img_feats.dim() == 2:
                # [B, D_img] -> [B, 1, hidden_size]
                img_emb = self.img_projector(img_feats).unsqueeze(1)
            else:
                # [B, N, D_img] -> [B, N, hidden_size]
                img_emb = self.img_projector(img_feats)

            # =============================
            # 3) Cross-Attention
            # =============================
            # 双向交叉注意力后，得到融合后的 text_seq, img_seq
            fused_text_seq, fused_img_seq = self.cross_attn(text_seq, img_emb)
            
            # 常见做法：取文本侧的 [CLS] 位置作为融合后的全局表示
            # BERT 一般第0个是 [CLS]
            fused_cls = fused_text_seq[:, 0]  # [B, hidden_size]

            # =============================
            # 4) 分类
            # =============================
            logits = self.classifier(fused_cls)

        else:  # late
            # =============================
            # 1) 文本 -> [CLS] -> text_logits
            # =============================
            text_emb = self.text_encoder(texts, return_cls_only=True)  # [B, hidden_size]
            text_logits = self.text_classifier(text_emb)  # [B, num_classes]

            # =============================
            # 2) 图像 -> 投影 -> image_logits
            # =============================
            if self.img_projector is None:
                in_dim = img_feats.shape[-1]
                out_dim = self.text_hidden_size
                self.img_projector = nn.Linear(in_dim, out_dim).to(device)
            if img_feats.dim() == 3:
                img_feats = img_feats.mean(dim=1)  # [B, D_img]
            img_emb = self.img_projector(img_feats)  # [B, hidden_size]

            if self.image_classifier is None:
                self.image_classifier = nn.Sequential(
                    nn.LayerNorm(self.text_hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.text_hidden_size, self.num_classes)
                ).to(device)
            img_logits = self.image_classifier(img_emb)   # [B, num_classes]

            # =============================
            # 3) late fusion: alpha 加权
            # =============================
            alpha = torch.sigmoid(self.late_weight)  # 标量 in (0,1)
            logits = alpha * text_logits + (1 - alpha) * img_logits

        return logits