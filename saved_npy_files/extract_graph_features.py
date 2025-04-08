#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

# 如果没有安装 PyTorch Geometric，请先安装: pip install torch-geometric -f https://data.pyg.org/whl/torch-xxx.html
try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv, Linear
except ImportError:
    raise ImportError("请先安装 PyTorch Geometric: pip install torch-geometric (以及相关依赖)")

from transformers import AutoTokenizer, AutoModel


##########################################################
# 1) Config
##########################################################
class Config:
    BASE_DIR = "E:/pythonCode/MR2_baseline"
    queries_root = os.path.join(BASE_DIR, "queries_dataset_merge")
    # 原始数据 JSON 路径
    dataset_train_json = os.path.join(queries_root, "dataset_items_train.json")
    dataset_val_json   = os.path.join(queries_root, "dataset_items_val.json")
    dataset_test_json  = os.path.join(queries_root, "dataset_items_test.json")

    # 输出的图向量保存目录
    saved_npy_folder = os.path.join(BASE_DIR, "saved_npy_files")
    os.makedirs(saved_npy_folder, exist_ok=True)

    # 训练的一些超参数
    GNN_EPOCHS   = 3
    HIDDEN_DIM   = 256          # 希望最终都变成 256 维以供 GNN 使用
    LR           = 1e-4
    WEIGHT_DECAY = 1e-4

    # 使用的文本模型 (xlm-roberta-large 的 hidden_size = 1024)
    TEXT_MODEL_NAME = "xlm-roberta-large"
    MAX_TEXT_LEN    = 128


##########################################################
# 2) 数据读取 / JSON 处理函数
##########################################################
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_direct_json(direct_path: str):
    """
    direct_annotation.json 的结构为：
    {
      "images_with_captions": [...],
      "images_with_no_captions": [...],
      "images_with_caption_matched_tags": [...]
    }
    """
    direct_json_path = os.path.join(direct_path, "direct_annotation.json")
    if os.path.isfile(direct_json_path):
        return load_json(direct_json_path)
    return {}

def load_inv_json(inv_path: str):
    """
    inverse_annotation.json 的结构为：
    {
       "entities": [...],
       "entities_scores": [...],
       "best_guess_lbl": [...],
       "all_fully_matched_captions": [...],
       "all_partially_matched_captions": [...],
       "partially_matched_no_text": [...],
       "fully_matched_no_text": [...]
    }
    """
    inv_json_path = os.path.join(inv_path, "inverse_annotation.json")
    if os.path.isfile(inv_json_path):
        return load_json(inv_json_path)
    return {}


##########################################################
# 3) 文本编码器: 用于对各种节点的文本进行编码 (XLM-R Large)
##########################################################
class TextEncoder(nn.Module):
    def __init__(self, model_name, max_length):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # xlm-roberta-large = 1024
        self.max_length = max_length

    def forward(self, text_list):
        """
        text_list: list of strings
        return: [batch_size, hidden_size=1024]
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text_list, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(device)
        outputs = self.model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb


##########################################################
# 4) 构建异质图（先只建立“正向”边）
##########################################################
def build_hetero_graph(train_dict, val_dict, test_dict):
    """
    构建一个包含 9 种节点类型的异质图，并将 claim 节点打上 label / split mask。
    最后仅返回已构造好的 HeteroData（不含任何特征），以及各种辅助映射。
    """

    # 合并为 all_claims
    all_claims = {}
    def add_split(dct, split_name):
        for k,v in dct.items():
            all_claims[f"{split_name}_{k}"] = v
            v["split"] = split_name

    add_split(train_dict, "train")
    add_split(val_dict,   "val")
    add_split(test_dict,  "test")

    # 记录去重后的节点
    domain2id   = {}
    page2id     = {}
    title2id    = {}
    snippet2id  = {}
    image2id    = {}
    alt2id      = {}
    entity2id   = {}
    bestGuess2id= {}

    # edge 列表
    edges_claim_page     = []
    edges_page_domain    = []
    edges_page_snippet   = []
    edges_page_title     = []
    edges_page_image     = []
    edges_page_alt       = []
    edges_claim_entity   = []
    edges_claim_bestGuess= []

    # 对 claim 分配索引
    claim_list = sorted(all_claims.keys())
    claim2id   = {cid: i for i, cid in enumerate(claim_list)}

    # 记录 claim 节点的标签/掩码
    num_claims = len(claim_list)
    labels     = -1 * torch.ones(num_claims, dtype=torch.long)
    train_mask = torch.zeros(num_claims, dtype=torch.bool)
    val_mask   = torch.zeros(num_claims, dtype=torch.bool)
    test_mask  = torch.zeros(num_claims, dtype=torch.bool)

    # ================ 遍历每条 Claim，抽取各种信息 ================
    for c_str in claim_list:
        item = all_claims[c_str]
        c_idx = claim2id[c_str]
        # label
        if "label" in item:
            labels[c_idx] = int(item["label"])
        # split mask
        sp = item["split"]
        if sp == "train":
            train_mask[c_idx] = True
        elif sp == "val":
            val_mask[c_idx]   = True
        else:
            test_mask[c_idx]  = True

        # 读取 direct_annotation.json
        direct_path = os.path.join(Config.queries_root, item["direct_path"])
        d_json = load_direct_json(direct_path)

        combined_blocks = []
        if "images_with_captions" in d_json:
            combined_blocks.extend(d_json["images_with_captions"])
        if "images_with_no_captions" in d_json:
            combined_blocks.extend(d_json["images_with_no_captions"])
        if "images_with_caption_matched_tags" in d_json:
            combined_blocks.extend(d_json["images_with_caption_matched_tags"])

        for block in combined_blocks:
            page_link = block.get("page_link", "").strip()
            domain_str= block.get("domain", "").strip()
            snippet_str = block.get("snippet", "").strip()
            page_title_str = block.get("page_title", "").strip()
            image_link_str = block.get("img_link", "").strip()

            # alt_node
            alt_node_str = ""
            cap_dict = block.get("caption", {})
            if isinstance(cap_dict, dict):
                alt_node_str = cap_dict.get("alt_node", "").strip()

            # claim -> page
            if page_link:
                if page_link not in page2id:
                    page2id[page_link] = len(page2id)
                p_idx = page2id[page_link]
                edges_claim_page.append((c_idx, p_idx))

            # page -> domain
            if page_link and domain_str:
                if domain_str not in domain2id:
                    domain2id[domain_str] = len(domain2id)
                d_idx = domain2id[domain_str]
                p_idx = page2id[page_link]
                edges_page_domain.append((p_idx, d_idx))

            # page -> snippet
            if page_link and snippet_str:
                if snippet_str not in snippet2id:
                    snippet2id[snippet_str] = len(snippet2id)
                s_idx = snippet2id[snippet_str]
                p_idx = page2id[page_link]
                edges_page_snippet.append((p_idx, s_idx))

            # page -> title
            if page_link and page_title_str:
                if page_title_str not in title2id:
                    title2id[page_title_str] = len(title2id)
                t_idx = title2id[page_title_str]
                p_idx = page2id[page_link]
                edges_page_title.append((p_idx, t_idx))

            # page -> image
            if page_link and image_link_str:
                if image_link_str not in image2id:
                    image2id[image_link_str] = len(image2id)
                i_idx = image2id[image_link_str]
                p_idx = page2id[page_link]
                edges_page_image.append((p_idx, i_idx))

            # page -> alt
            if page_link and alt_node_str:
                if alt_node_str not in alt2id:
                    alt2id[alt_node_str] = len(alt2id)
                a_idx = alt2id[alt_node_str]
                p_idx = page2id[page_link]
                edges_page_alt.append((p_idx, a_idx))

        # ============= inverse_annotation ==============
        inv_path = os.path.join(Config.queries_root, item["inv_path"])
        inv_json = load_inv_json(inv_path)

        # claim -> entity
        ents = inv_json.get("entities", [])
        for e in ents:
            e_str = e.strip()
            if e_str:
                if e_str not in entity2id:
                    entity2id[e_str] = len(entity2id)
                e_idx = entity2id[e_str]
                edges_claim_entity.append((c_idx, e_idx))

        # claim -> best_guess
        best_list = inv_json.get("best_guess_lbl", [])
        for b in best_list:
            b_str = b.strip()
            if b_str:
                if b_str not in bestGuess2id:
                    bestGuess2id[b_str] = len(bestGuess2id)
                b_idx = bestGuess2id[b_str]
                edges_claim_bestGuess.append((c_idx, b_idx))

        # 处理其他 matched
        match_fields = [
            "all_fully_matched_captions",
            "all_partially_matched_captions",
            "partially_matched_no_text",
            "fully_matched_no_text"
        ]
        matched_blocks = []
        for mf in match_fields:
            if mf in inv_json and isinstance(inv_json[mf], list):
                matched_blocks.extend(inv_json[mf])

        for block in matched_blocks:
            page_link = block.get("page_link","").strip()
            title_str = block.get("title","").strip()
            image_link_str = block.get("image_link","").strip()

            cdict = block.get("caption", {})
            alt_node_str = ""
            if isinstance(cdict, dict):
                alt_node_str = cdict.get("title_node", "").strip()

            # claim -> page
            if page_link:
                if page_link not in page2id:
                    page2id[page_link] = len(page2id)
                p_idx = page2id[page_link]
                edges_claim_page.append((c_idx, p_idx))

            # page -> title
            if page_link and title_str:
                if title_str not in title2id:
                    title2id[title_str] = len(title2id)
                t_idx = title2id[title_str]
                p_idx = page2id[page_link]
                edges_page_title.append((p_idx, t_idx))

            # page -> image
            if page_link and image_link_str:
                if image_link_str not in image2id:
                    image2id[image_link_str] = len(image2id)
                i_idx = image2id[image_link_str]
                p_idx = page2id[page_link]
                edges_page_image.append((p_idx, i_idx))

            # page -> alt
            if page_link and alt_node_str:
                if alt_node_str not in alt2id:
                    alt2id[alt_node_str] = len(alt2id)
                a_idx = alt2id[alt_node_str]
                p_idx = page2id[page_link]
                edges_page_alt.append((p_idx, a_idx))

    # ============ 统计各类节点数量 =============
    num_claim  = len(claim_list)
    num_domain = len(domain2id)
    num_page   = len(page2id)
    num_title  = len(title2id)
    num_snippet= len(snippet2id)
    num_image  = len(image2id)
    num_alt    = len(alt2id)
    num_entity = len(entity2id)
    num_best   = len(bestGuess2id)

    print(f"[Info] #Claim={num_claim}, #Domain={num_domain}, #Page={num_page}, #Title={num_title}, "
          f"#Snippet={num_snippet}, #Image={num_image}, #Alt={num_alt}, "
          f"#Entity={num_entity}, #BestGuess={num_best}")

    # ============ 创建 HeteroData 实例 & 分配空 x ============
    data = HeteroData()

    # claim
    data["claim"].x = torch.zeros(num_claim, Config.HIDDEN_DIM)  # shape=[num_claim, 256]
    data["claim"].y = labels
    data["claim"].train_mask = train_mask
    data["claim"].val_mask   = val_mask
    data["claim"].test_mask  = test_mask

    # 其他节点 (先都用 zeros；稍后会用文本投影去替换)
    data["domain"].x     = torch.zeros(num_domain, Config.HIDDEN_DIM)
    data["page"].x       = torch.zeros(num_page,   Config.HIDDEN_DIM)
    data["title"].x      = torch.zeros(num_title,  Config.HIDDEN_DIM)
    data["snippet"].x    = torch.zeros(num_snippet,Config.HIDDEN_DIM)
    data["image"].x      = torch.zeros(num_image,  Config.HIDDEN_DIM)
    data["alt"].x        = torch.zeros(num_alt,    Config.HIDDEN_DIM)
    data["entity"].x     = torch.zeros(num_entity, Config.HIDDEN_DIM)
    data["best_guess"].x = torch.zeros(num_best,   Config.HIDDEN_DIM)

    # ============ 建立边 ============
    def add_edge_index(edge_tuples, src_t, rel, dst_t):
        if not edge_tuples:
            return
        arr = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
        data[(src_t, rel, dst_t)].edge_index = arr

    # claim->page
    add_edge_index(edges_claim_page,     "claim", "to_page",      "page")
    # page->domain
    add_edge_index(edges_page_domain,    "page",  "to_domain",    "domain")
    # page->snippet
    add_edge_index(edges_page_snippet,   "page",  "to_snippet",   "snippet")
    # page->title
    add_edge_index(edges_page_title,     "page",  "to_title",     "title")
    # page->image
    add_edge_index(edges_page_image,     "page",  "to_image",     "image")
    # page->alt
    add_edge_index(edges_page_alt,       "page",  "to_alt",       "alt")
    # claim->entity
    add_edge_index(edges_claim_entity,   "claim", "to_entity",    "entity")
    # claim->best_guess
    add_edge_index(edges_claim_bestGuess,"claim", "to_best_guess","best_guess")

    return data, claim2id, domain2id, page2id, title2id, snippet2id, \
           image2id, alt2id, entity2id, bestGuess2id, claim_list


##########################################################
# (重要) 额外步骤：手动添加反向边
##########################################################
def add_reverse_edges(data: HeteroData, suffix="_rev"):
    """
    扫描 data 中已有的 (src_type, rel, dst_type)，
    为其自动添加 (dst_type, rel+suffix, src_type)，
    并将 edge_index 行维度翻转。
    """
    from copy import deepcopy
    old_edge_types = list(data.edge_types)  # PyG2.1 中是个 set-like，但用 list 更保险

    new_edges = []
    for (src, rel, dst) in old_edge_types:
        rev_rel = rel + suffix
        # 如果反向边还不存在，就添加
        if (dst, rev_rel, src) not in data.edge_types:
            edge_idx = data[(src, rel, dst)].edge_index
            # 形状 [2, E]，翻转行维度即可得到反向边
            rev_idx = edge_idx[[1, 0], :]
            new_edges.append(((dst, rev_rel, src), rev_idx))

    # 正式加入 data
    for (edge_type, rev_idx) in new_edges:
        data[edge_type].edge_index = rev_idx

    return data


##########################################################
# 5) 为各种类型节点建立文本特征 (含线性投影到 256)
##########################################################
def build_node_features(data, claim_list,
                        domain2id, page2id, title2id, snippet2id,
                        image2id, alt2id, entity2id, bestGuess2id):
    """
    使用 XLM-R Large 对文本节点进行编码，再投影到 256 维。
    其中，claim 本身如果要用文本嵌入，也可以在此处理。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 初始化文本编码器 (XLM-R Large)
    encoder = TextEncoder(Config.TEXT_MODEL_NAME, Config.MAX_TEXT_LEN).to(device)
    encoder.eval()

    # 2) 建立投影层: 1024 -> 256
    projector = nn.Linear(encoder.hidden_size, Config.HIDDEN_DIM, bias=False).to(device)
    projector.eval()

    # 3) 定义一个批量编码并投影的函数
    def encode_text_batch(strings):
        out_embs = []
        bs = 16  # batch size
        with torch.no_grad():
            for i in range(0, len(strings), bs):
                batch_s = strings[i:i+bs]
                emb_1024 = encoder(batch_s)     # [b, 1024]
                emb_256  = projector(emb_1024)  # [b, 256]
                out_embs.append(emb_256.cpu())
        if out_embs:
            return torch.cat(out_embs, dim=0)  # [N, 256]
        else:
            return torch.zeros(0, Config.HIDDEN_DIM)

    # 4) 对各类型节点的文本做编码 + 投影

    # domain
    domain_list = sorted(domain2id.keys(), key=lambda x: domain2id[x])
    if domain_list:
        domain_embs = encode_text_batch(domain_list)
        data["domain"].x = domain_embs

    # page
    page_list = sorted(page2id.keys(), key=lambda x: page2id[x])
    if page_list:
        page_embs = encode_text_batch(page_list)
        data["page"].x = page_embs

    # snippet
    snippet_list = sorted(snippet2id.keys(), key=lambda x: snippet2id[x])
    if snippet_list:
        snippet_embs = encode_text_batch(snippet_list)
        data["snippet"].x = snippet_embs

    # title
    title_list = sorted(title2id.keys(), key=lambda x: title2id[x])
    if title_list:
        title_embs = encode_text_batch(title_list)
        data["title"].x = title_embs

    # image
    image_list = sorted(image2id.keys(), key=lambda x: image2id[x])
    if image_list:
        image_embs = encode_text_batch(image_list)
        data["image"].x = image_embs

    # alt
    alt_list = sorted(alt2id.keys(), key=lambda x: alt2id[x])
    if alt_list:
        alt_embs = encode_text_batch(alt_list)
        data["alt"].x = alt_embs

    # entity
    entity_list = sorted(entity2id.keys(), key=lambda x: entity2id[x])
    if entity_list:
        entity_embs = encode_text_batch(entity_list)
        data["entity"].x = entity_embs

    # best_guess
    best_list = sorted(bestGuess2id.keys(), key=lambda x: bestGuess2id[x])
    if best_list:
        best_embs = encode_text_batch(best_list)
        data["best_guess"].x = best_embs

    return data


##########################################################
# 6) 简单示例: 定义一个2层 HGTConv 模型，对 claim 做三分类
##########################################################
class BigHeteroGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=3, metadata=None):
        super().__init__()
        self.conv1 = HGTConv(
            in_channels=in_dim, 
            out_channels=hidden_dim,
            metadata=metadata, 
            heads=4
        )
        self.conv2 = HGTConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,
            metadata=metadata, 
            heads=4
        )
        self.lin = Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        # 先做第一层
        x_dict = self.conv1(x_dict, edge_index_dict)
        for k in x_dict:
            x_dict[k] = F.relu(x_dict[k])
        # 第二层
        x_dict = self.conv2(x_dict, edge_index_dict)
        for k in x_dict:
            x_dict[k] = F.relu(x_dict[k])
        # 对 claim 节点做分类
        out = self.lin(x_dict["claim"])
        return out, x_dict


##########################################################
# 7) 训练
##########################################################
def train_graph_model(data: HeteroData):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_types = list(data.node_types)
    edge_types = list(data.edge_types)

    # 这里 in_dim = 256
    in_dim = data["claim"].x.size(1)
    model = BigHeteroGNN(
        in_dim=in_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=3,
        metadata=(node_types, edge_types)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    data = data.to(device)

    y       = data["claim"].y
    tmask   = data["claim"].train_mask
    vmask   = data["claim"].val_mask

    for epoch in range(Config.GNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(logits[tmask], y[tmask])
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(data.x_dict, data.edge_index_dict)
            val_pred = val_logits[vmask].argmax(dim=1)
            val_acc = (val_pred == y[vmask]).float().mean().item()
        print(f"[Epoch {epoch+1}/{Config.GNN_EPOCHS}] train_loss={loss.item():.4f}, val_acc={val_acc:.4f}")

    # 最终输出 claim 节点表示
    model.eval()
    with torch.no_grad():
        _, x_dict = model(data.x_dict, data.edge_index_dict)
        claim_emb = x_dict["claim"].cpu().numpy()
    return claim_emb


##########################################################
# 8) 保存 embeddings
##########################################################
def save_claim_embeddings(claim_embs, claim_list, prefix="graph_features"):
    """
    claim_embs: [num_claim, hidden_dim]
    claim_list: ["train_0","train_1",...,"val_5","test_2",...]
    """
    train_dict, val_dict, test_dict = {}, {}, {}
    for i, c_id in enumerate(claim_list):
        sp, idx_str = c_id.split("_", 1)
        emb = claim_embs[i]
        if sp == "train":
            train_dict[idx_str] = emb
        elif sp == "val":
            val_dict[idx_str]   = emb
        else:
            test_dict[idx_str]  = emb

    train_path = os.path.join(Config.saved_npy_folder, f"{prefix}_train.npy")
    val_path   = os.path.join(Config.saved_npy_folder, f"{prefix}_val.npy")
    test_path  = os.path.join(Config.saved_npy_folder, f"{prefix}_test.npy")
    np.save(train_path, train_dict)
    np.save(val_path,   val_dict)
    np.save(test_path,  test_dict)
    print(f"[Done] Saved embeddings:\n{train_path}\n{val_path}\n{test_path}")


##########################################################
# 9) 主函数
##########################################################
def main():
    # 1) 加载数据
    train_data = load_json(Config.dataset_train_json)
    val_data   = load_json(Config.dataset_val_json)
    test_data  = load_json(Config.dataset_test_json)

    # 2) 构建更复杂的异质图 (此时尚未添加反向边)
    graph_data, claim2id, domain2id, page2id, title2id, snippet2id, \
        image2id, alt2id, entity2id, bestGuess2id, claim_list = build_hetero_graph(
            train_data, val_data, test_data
        )

    # 2.1) 关键：手动添加反向边，避免 HGTConv 缺少 src->dst 或 dst->src
    graph_data = add_reverse_edges(graph_data)

    # 3) 构建节点特征 (包含 XLM-R 编码 + 线性投影)
    graph_data = build_node_features(
        graph_data, claim_list,
        domain2id, page2id, title2id, snippet2id,
        image2id, alt2id, entity2id, bestGuess2id
    )

    # （可选）如果你想对 claim 节点文本也做类似编码，可在此处处理
    # 默认为 claim.x 全零，示例中只利用 GNN 结构本身做信息聚合

    # 4) 训练 GNN
    claim_embs = train_graph_model(graph_data)

    # 5) 保存
    save_claim_embeddings(claim_embs, claim_list, prefix="graph_features")


if __name__ == "__main__":
    main()