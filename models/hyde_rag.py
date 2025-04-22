#!/usr/bin/python
# -*- coding:utf-8 -*-
# Author: sqhyz55
# @Time : 2025/4/22

import os
import json
from typing import TypedDict, List
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise ValueError("请设置 .env 文件或环境变量 ARK_API_KEY")

# 初始化火山引擎 OpenAI 客户端
client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

# === 加载上下文数据 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "hotpot_flat_contexts.json")

with open(data_path, "r") as f:
    context_data = json.load(f)

corpus = [item["text"] for item in context_data]
vectorizer = TfidfVectorizer().fit(corpus)
corpus_vectors = vectorizer.transform(corpus)


# === 定义状态结构 ===
class MyState(TypedDict):
    question: str
    hypo_answer: str
    retrieved_context: List[str]
    answer: str


# === Graph 节点函数定义 ===

def input_node(state: MyState) -> MyState:
    print(f"\n[用户问题]{state['question']}")
    return state


def hypo_gen_node(state: MyState) -> MyState:
    question = state["question"]
    prompt = f"请根据以下问题生成一个合理的假设回答，不需要保证准确，只需提供有信息量的内容用于文档检索：\n\n问题：{question}\n\n假设回答："

    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": prompt},
        ]
    )
    hypo_answer = response.choices[0].message.content
    state["hypo_answer"] = hypo_answer
    print(f"\n[假设回答用于检索]{hypo_answer}")
    return state


def retriever_node(state: MyState) -> MyState:
    hypo = state["hypo_answer"]
    hypo_vec = vectorizer.transform([hypo])
    scores = cosine_similarity(hypo_vec, corpus_vectors).flatten()
    top_indices = scores.argsort()[-5:][::-1]
    retrieved = [corpus[i] for i in top_indices]
    state["retrieved_context"] = retrieved
    print("\n[检索结果]")
    for i, sent in enumerate(retrieved, 1):
        print(f"{i}. {sent}")
    return state


def answer_node(state: MyState) -> MyState:
    context = "\n".join(state["retrieved_context"])
    question = state["question"]
    prompt = f"""
你是一名知识问答助手，请根据以下背景信息回答问题。

背景：
{context}

问题：
{question}

请简洁准确地作答：
"""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是人工智能问答助手"},
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content
    print(f"\n[最终回答]{answer}")
    state["answer"] = answer
    return state


# === 构建 LangGraph ===
workflow = StateGraph(state_schema=MyState)

workflow.add_node("input", input_node)
workflow.add_node("generate_hypo", hypo_gen_node)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("generate", answer_node)

workflow.set_entry_point("input")
workflow.add_edge("input", "generate_hypo")
workflow.add_edge("generate_hypo", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# === 交互式运行 ===
if __name__ == "__main__":
    while True:
        question = input("\n请输入一个问题（或输入 exit 退出）:\n> ")
        if question.lower() in {"exit", "quit"}:
            break
        app.invoke({"question": question})
