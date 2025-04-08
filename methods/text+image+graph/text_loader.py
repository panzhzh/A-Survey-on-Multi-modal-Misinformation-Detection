#!/usr/bin/env python3
# text_loader.py
# -*- coding: utf-8 -*-

import os
import json


def process_string(text: str) -> str:
    """
    对文本做简单清洗。
    """
    return (text
            .replace('&#39;', ' ')
            .replace('<b>', '')
            .replace('</b>', '')
            .strip())


def load_inv_json(inv_path: str) -> dict:
    """
    读取 inverse_annotation.json 并返回其内容（字典）。
    若不存在或解析失败，返回空字典。
    """
    inv_json_path = os.path.join(inv_path, "inverse_annotation.json")
    if os.path.isfile(inv_json_path):
        try:
            with open(inv_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def load_direct_json(direct_path: str) -> list:
    """
    读取 direct_annotation.json 并返回其内容（列表）。
    若不存在或解析失败，返回空列表。
    """
    direct_json_path = os.path.join(direct_path, "direct_annotation.json")
    if os.path.isfile(direct_json_path):
        try:
            with open(direct_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except:
            pass
    return []


def build_text(caption: str, inv_data: dict, direct_data: list) -> str:
    """
    将 caption + inverse_annotation + direct_annotation 内容全部拼接到一起，
    得到一个完整文本并返回（相当于原先的 all_text）。
    """
    # 1) Caption
    caption_clean = process_string(caption)
    final_text = f"[CAP] {caption_clean}"

    # 2) inverse_annotation 里的实体 (entities / best_guess_lbl) + matched 信息
    #    相当于之前的 [ENT] 与 [MAT] 部分，这里直接写死阈值 0.6
    if "entities" in inv_data and "entities_scores" in inv_data:
        ents = inv_data["entities"]
        ents_scores = inv_data["entities_scores"]
        if len(ents) == len(ents_scores):
            ent_str_list = []
            for e, s in zip(ents, ents_scores):
                if s >= 0.6:  # 阈值固定
                    e_clean = process_string(e)
                    ent_str_list.append(f"{e_clean} [SCORE={s:.3f}]")

            # best_guess_lbl
            if "best_guess_lbl" in inv_data and isinstance(inv_data["best_guess_lbl"], list):
                for g in inv_data["best_guess_lbl"]:
                    g_clean = process_string(g)
                    ent_str_list.append(f"{g_clean} [BEST_GUESS]")

            if len(ent_str_list) > 0:
                merged_ents = " <SEP> ".join(ent_str_list)
                final_text += f" [ENT] {merged_ents}"

    # 3) matched 信息
    fields_to_collect = [
        "all_fully_matched_captions",
        "all_partially_matched_captions",
        "partially_matched_no_text",
        "fully_matched_no_text"
    ]
    matched_strings = []
    for field in fields_to_collect:
        if field in inv_data and isinstance(inv_data[field], list):
            for block in inv_data[field]:
                t = process_string(block.get("title", ""))
                d = process_string(block.get("domain", ""))
                c = process_string(block.get("content", ""))
                pieces = [x for x in [t, d, c] if x]
                if pieces:
                    matched_strings.append(" / ".join(pieces))

    if matched_strings:
        final_text += " [MAT] " + " <SEP> ".join(matched_strings)

    # 4) direct_annotation 信息 (page_title / domain / content)
    #    相当于之前的 [EVID] 部分
    items = []
    for block in direct_data:
        p = process_string(block.get("page_title", ""))
        d = process_string(block.get("domain", ""))
        c = process_string(block.get("content", ""))
        sub_pieces = [x for x in [p, d, c] if x]
        if sub_pieces:
            combined = " / ".join(sub_pieces)
            items.append(combined)

    if items:
        final_text += " [EVID] " + " <SEP> ".join(items)

    return final_text