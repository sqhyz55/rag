#!/usr/bin/python
# -*- coding:utf-8 -*-
# Author: sqhyz55
# @Time : 2025/4/22
import json
import os
import argparse


def load_hotpot_contexts(file_path, max_items=1000):
    with open(file_path, 'r') as f:
        data = json.load(f)

    docs = []
    for item in data[:max_items]:
        for title, sentences in item['context']:
            for sent in sentences:
                docs.append({"title": title, "text": sent})
    return docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten HotpotQA contexts.")
    parser.add_argument('--input', type=str, default=None, help="Path to input JSON file")
    parser.add_argument('--output', type=str, default=None, help="Path to output JSON file")
    parser.add_argument('--max_items', type=int, default=1000, help="Max items to process")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置默认路径
    default_input = os.path.join(script_dir, "..", "data", "hotpot_train_v1.1.json")
    default_output = os.path.join(script_dir, "..", "data", "hotpot_flat_contexts.json")

    in_path = args.input if args.input else default_input
    out_path = args.output if args.output else default_output

    contexts = load_hotpot_contexts(in_path, max_items=args.max_items)

    with open(out_path, 'w') as f:
        json.dump(contexts, f, indent=2)

    print(f"保存{len(contexts)} 条支持句到 {out_path}")

