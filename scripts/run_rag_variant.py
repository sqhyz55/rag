#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: sqhyz55
# @Time : 2025/4/22

import argparse
import importlib

# 支持的模型映射关系：键是命令行参数名，值是模块路径
RAG_VARIANTS = {
    "basic": "rag_base.basic_rag",
    "corrective": "models.corrective_rag",
    "fusion": "models.rag_fusion",
    "hyde": "models.hyde",
    "raptor": "models.raptor"
}

def main():
    parser = argparse.ArgumentParser(description="运行不同的 RAG 模型实现")
    parser.add_argument(
        "--variant", "-v",
        choices=RAG_VARIANTS.keys(),
        default="basic",
        help="选择要运行的 RAG 模型变体，如 basic、corrective、fusion、hyde、raptor"
    )

    args = parser.parse_args()
    module_path = RAG_VARIANTS[args.variant]

    print(f"👉 正在加载模型: {args.variant} ({module_path})")

    try:
        rag_module = importlib.import_module(module_path)
        if hasattr(rag_module, "app"):
            while True:
                question = input("\n请输入一个问题（或者输入 exit 退出）: \n> ")
                if question.lower() in {"exit", "quit"}:
                    break
                rag_module.app.invoke({"question": question})
        else:
            raise AttributeError("模块中未找到 'app' 实例")
    except Exception as e:
        print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    main()
