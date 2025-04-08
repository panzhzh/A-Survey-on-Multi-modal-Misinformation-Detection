#!/usr/bin/env python3
# model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# 这里我们改用 CLIP 相关的 HuggingFace 接口
from transformers import CLIPModel, CLIPTokenizer


##############################################################################
# 1) CLIP 文本编码器
##############################################################################
class ClipTextEncoder(nn.Module):
    def __init__(self, pretrained_name="openai/clip-vit-large-patch14", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_name)
        self.model = CLIPModel.from_pretrained(pretrained_name)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.max_length = max_length

    def forward(self, list_of_texts, return_cls_only=True):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            list_of_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)

        text_embeds = self.model.get_text_features(**inputs)  # shape [B, hidden_size]

        if return_cls_only:
            # Early / Late fusion时，需要的是 [B, d]
            return text_embeds
        else:
            # Mid-fusion时，需要序列维度，这里简单地把 [B, d] -> [B, 1, d]
            return text_embeds.unsqueeze(1)

##############################################################################
# 2) 自定义损失（BalancedCrossEntropyLoss / FocalLoss），和以前保持一致
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
# 3) GatedFusion (若有需要中期融合的示例，可保留)
##############################################################################
class GatedFusion(nn.Module):
    """
    简单示例：在文本向量和图像向量拼接前，先用 gating 进行融合。
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
# 4) CrossAttentionFusion (若要更成熟的中期融合，可保留该模块)
##############################################################################
class CrossAttentionFusion(nn.Module):
    """
    小型双向 Cross-Attention，用于文本序列和图像序列的交互。
    如果图像只有一个全局向量，可视为 I=1。
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn_t2i = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_i2t = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln_text = nn.LayerNorm(d_model)
        self.ln_img  = nn.LayerNorm(d_model)

    def forward(self, text_emb, img_emb, text_mask=None, img_mask=None):
        """
        Args:
            text_emb: [B, L, d]
            img_emb:  [B, I, d]
        Return:
            text_out: [B, L, d]
            img_out:  [B, I, d]
        """
        # 1) 以文本为 Q，看图像 K/V
        t2i_out, _ = self.cross_attn_t2i(
            query=text_emb,
            key=img_emb,
            value=img_emb,
            key_padding_mask=img_mask  # 可选
        )
        text_out = self.ln_text(text_emb + t2i_out)

        # 2) 以图像为 Q，看文本 K/V
        i2t_out, _ = self.cross_attn_i2t(
            query=img_emb,
            key=text_emb,
            value=text_emb,
            key_padding_mask=text_mask  # 可选
        )
        img_out = self.ln_img(img_emb + i2t_out)

        return text_out, img_out


##############################################################################
# 5) 多模态模型：我们只保留 CLIP 的文本编码器 + 离线图像特征
#    并支持 early/mid/late 三种融合方式作为示例
##############################################################################
class MultiModalNetwork(nn.Module):
    def __init__(
        self,
        fusion_level="early",
        num_classes=3,
        num_heads=4
    ):
        """
        仅保留 CLIP 逻辑：文本部分用 ClipTextEncoder，
        图像部分使用你离线提取的 CLIP 特征。
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_level = fusion_level.lower()

        # 初始化 CLIP 文本编码器
        self.text_encoder = ClipTextEncoder(pretrained_name="openai/clip-vit-large-patch14")
        self.text_hidden_size = self.text_encoder.hidden_size

        # 图像投影层（可能在 forward() 中动态创建）
        self.img_projector = None

        # 根据融合方式来构建对应的分类头
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
            self.cross_attn = CrossAttentionFusion(
                d_model=self.text_hidden_size,
                n_heads=num_heads
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size, num_classes)
            )

        elif self.fusion_level == "late":
            # 文本 logits & 图像 logits -> α 加权
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
          img_feats: Tensor[B, D] 或 [B, N, D] 的图像特征（已用 CLIP 离线提取）
        Returns:
          logits: [B, num_classes]
        """
        device = next(self.parameters()).device

        # ============== 不同融合方式 ==============
        if self.fusion_level == "early":
            # 1) 文本 -> [cls]
            text_emb = self.text_encoder(texts, return_cls_only=True)  # [B, hidden_size]

            # 2) 若必要，构建投影层
            if self.img_projector is None:
                in_dim = img_feats.shape[-1]
                self.img_projector = nn.Linear(in_dim, self.text_hidden_size).to(device)

            # 如果是 [B, N, D]，先做个 mean pooling -> [B, D]
            if img_feats.dim() == 3:
                img_feats = img_feats.mean(dim=1)
            img_emb = self.img_projector(img_feats)  # [B, hidden_size]

            # 3) 拼接 -> 分类
            fuse = torch.cat([text_emb, img_emb], dim=1)  # [B, 2*hidden_size]
            logits = self.classifier(fuse)

        elif self.fusion_level == "mid":
            # 1) 文本全序列（为了 cross-attn 演示，返回的其实是 [B,1,hidden], 仅供示例）
            text_seq = self.text_encoder(texts, return_cls_only=False)  # [B,1,hidden]

            # 2) 图像 -> 投影 -> [B, I, hidden]
            if self.img_projector is None:
                in_dim = img_feats.shape[-1]
                self.img_projector = nn.Linear(in_dim, self.text_hidden_size).to(device)

            if img_feats.dim() == 2:
                # [B,D] -> [B,1,D]
                img_emb = self.img_projector(img_feats).unsqueeze(1)
            else:
                # [B,N,D] -> [B,N,hidden]
                img_emb = self.img_projector(img_feats)

            # 3) Cross-Attention
            fused_text_seq, fused_img_seq = self.cross_attn(text_seq, img_emb)
            # 这里取文本侧第 0 个位置作为全局向量
            fused_cls = fused_text_seq[:, 0]  # [B, hidden]

            # 4) 分类
            logits = self.classifier(fused_cls)

        else:  # late
            # 1) 文本 -> text_logits
            text_emb = self.text_encoder(texts, return_cls_only=True)  # [B, hidden]
            text_logits = self._build_text_logits(text_emb)

            # 2) 图像 -> image_logits
            if self.img_projector is None:
                in_dim = img_feats.shape[-1]
                self.img_projector = nn.Linear(in_dim, self.text_hidden_size).to(device)
            if img_feats.dim() == 3:
                img_feats = img_feats.mean(dim=1)
            img_emb = self.img_projector(img_feats)  # [B, hidden]
            img_logits = self._build_image_logits(img_emb)

            # 3) α 加权融合
            alpha = torch.sigmoid(self.late_weight)
            logits = alpha * text_logits + (1 - alpha) * img_logits

        return logits

    def _build_text_logits(self, text_emb):
        if not hasattr(self, "text_classifier"):
            raise RuntimeError("Late-fusion mode but text_classifier not defined!")
        return self.text_classifier(text_emb)

    def _build_image_logits(self, img_emb):
        if self.image_classifier is None:
            self.image_classifier = nn.Sequential(
                nn.LayerNorm(self.text_hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.text_hidden_size, self.num_classes)
            ).to(img_emb.device)
        return self.image_classifier(img_emb)
