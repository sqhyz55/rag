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

# 加载环境变量
load_dotenv()

api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise ValueError("请设置 .env 文件或环境变量 ARK_API_KEY")

client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

# === 加载数据 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "hotpot_flat_contexts.json")

with open(data_path, "r") as f:
    context_data = json.load(f)

corpus = [item["text"] for item in context_data]
vectorizer = TfidfVectorizer().fit(corpus)
corpus_vectors = vectorizer.transform(corpus)

# === 状态结构 ===
class MyState(TypedDict):
    question: str
    subquestions: List[str]
    retrieved_context: List[str]
    answer: str

# === 节点函数 ===

def input_node(state: MyState) -> MyState:
    print(f"\n[用户问题]{state['question']}")
    return state

def decompose_node(state: MyState) -> MyState:
    question = state["question"]
    prompt = f"""
请将以下复杂问题分解为多个简单的子问题，用于信息检索：
问题：{question}

请输出子问题列表（每行一个）：
"""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是一个能拆解复杂问题的助手"},
            {"role": "user", "content": prompt},
        ]
    )
    lines = response.choices[0].message.content.strip().split("\n")
    subquestions = [line.strip(" -•：:") for line in lines if line.strip()]
    state["subquestions"] = subquestions
    print(f"\n[子问题列表]")
    for sq in subquestions:
        print(f"- {sq}")
    return state

def retrieve_node(state: MyState) -> MyState:
    subquestions = state["subquestions"]
    all_retrieved = []

    for sq in subquestions:
        sq_vec = vectorizer.transform([sq])
        scores = cosine_similarity(sq_vec, corpus_vectors).flatten()
        top_indices = scores.argsort()[-3:][::-1]
        retrieved = [corpus[i] for i in top_indices]
        all_retrieved.extend(retrieved)
        print(f"\n[为子问题: {sq} 检索结果]")
        for i, s in enumerate(retrieved, 1):
            print(f"{i}. {s}")

    # 去重
    state["retrieved_context"] = list(dict.fromkeys(all_retrieved))
    return state

def answer_node(state: MyState) -> MyState:
    context = "\n".join(state["retrieved_context"])
    question = state["question"]
    prompt = f"""
你是一名知识问答助手，请根据以下背景信息回答复杂问题。

背景：
{context}

问题：
{question}

请准确作答：
"""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是人工智能问答助手"},
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content
    print(f"\n[模型回答]{answer}")
    state["answer"] = answer
    return state

# === 构建 LangGraph ===
workflow = StateGraph(state_schema=MyState)

workflow.add_node("input", input_node)
workflow.add_node("decompose", decompose_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", answer_node)

workflow.set_entry_point("input")
workflow.add_edge("input", "decompose")
workflow.add_edge("decompose", "retrieve")
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
