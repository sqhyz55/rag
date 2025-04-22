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

# === 环境变量 ===
load_dotenv()
api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise ValueError("请设置 .env 文件或环境变量 ARK_API_KEY")

client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

# === 数据路径 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "hotpot_flat_contexts.json")

with open(data_path, "r") as f:
    context_data = json.load(f)

corpus = [item["text"] for item in context_data]
vectorizer = TfidfVectorizer().fit(corpus)
corpus_vectors = vectorizer.transform(corpus)

# === 状态定义 ===
class MyState(TypedDict):
    question: str
    retrieved_contexts: List[List[str]]
    fused_prompt: str
    answer: str

# === 节点定义 ===
def input_node(state: MyState) -> MyState:
    print(f"\n[用户问题] {state['question']}")
    return state

def multi_retriever_node(state: MyState) -> MyState:
    question = state["question"]
    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, corpus_vectors).flatten()
    top_indices = scores.argsort()[-15:][::-1]  # 选前15条

    retrieved_contexts = []
    for i in range(0, 15, 5):
        chunk = [corpus[idx] for idx in top_indices[i:i+5]]
        retrieved_contexts.append(chunk)

    print("\n[多轮检索结果]")
    for i, ctxs in enumerate(retrieved_contexts):
        print(f"\n-- 批次 {i+1} --")
        for j, c in enumerate(ctxs, 1):
            print(f"{j}. {c}")

    state["retrieved_contexts"] = retrieved_contexts
    return state

def fusion_node(state: MyState) -> MyState:
    question = state["question"]
    prompt_blocks = []
    for i, ctxs in enumerate(state["retrieved_contexts"]):
        context = "\n".join(ctxs)
        prompt = f"批次 {i+1}：\n背景：\n{context}\n问题：{question}"
        prompt_blocks.append(prompt)

    fused_prompt = "\n\n".join(prompt_blocks) + "\n\n请综合以上多个背景信息，准确简洁地回答问题。"
    state["fused_prompt"] = fused_prompt
    return state

def answer_node(state: MyState) -> MyState:
    prompt = state["fused_prompt"]

    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是人工智能问答助手"},
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content
    print(f"\n[模型最终回答] {answer}")
    state["answer"] = answer
    return state

# === 构建LangGraph ===
workflow = StateGraph(state_schema=MyState)
workflow.add_node("input", input_node)
workflow.add_node("retrieve", multi_retriever_node)
workflow.add_node("fuse", fusion_node)
workflow.add_node("generate", answer_node)

workflow.set_entry_point("input")
workflow.add_edge("input", "retrieve")
workflow.add_edge("retrieve", "fuse")
workflow.add_edge("fuse", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# === 交互式运行 ===
if __name__ == "__main__":
    while True:
        question = input("\n请输入一个问题（或者输入 exit 退出）: \n> ")
        if question.lower() in {"exit", "quit"}:
            break
        app.invoke({"question": question})
