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

# === 加载环境变量 ===
load_dotenv()
api_key = os.getenv("ARK_API_KEY")
if not api_key:
    raise ValueError("请设置 .env 文件或环境变量 ARK_API_KEY")

# 初始化客户端
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
class CorrectiveState(TypedDict):
    question: str
    first_answer: str
    retrieved_context: List[str]
    corrected_answer: str

# === 节点 ===
def input_node(state: CorrectiveState) -> CorrectiveState:
    print(f"\n[用户问题]{state['question']}")
    return state

def initial_retrieve_and_answer(state: CorrectiveState) -> CorrectiveState:
    question = state["question"]
    vec = vectorizer.transform([question])
    sims = cosine_similarity(vec, corpus_vectors).flatten()
    top_indices = sims.argsort()[-5:][::-1]
    retrieved = [corpus[i] for i in top_indices]

    context = "\n".join(retrieved)
    prompt = f"""
你是一名问答助手，请根据背景信息回答用户问题：

背景：
{context}

问题：
{question}

请作答：
"""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是人工智能问答助手"},
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content
    print(f"\n[初始回答]{answer}")
    state["retrieved_context"] = retrieved
    state["first_answer"] = answer
    return state

def corrective_retrieve_and_answer(state: CorrectiveState) -> CorrectiveState:
    first_answer = state["first_answer"]
    vec = vectorizer.transform([first_answer])
    sims = cosine_similarity(vec, corpus_vectors).flatten()
    top_indices = sims.argsort()[-5:][::-1]
    retrieved = [corpus[i] for i in top_indices]

    context = "\n".join(retrieved)
    prompt = f"""
你是一名问答助手，以下是原始问题和一个初始回答，请根据新背景进行修正：

原问题：
{state['question']}

初始回答：
{first_answer}

补充背景：
{context}

请给出更准确的回答：
"""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是人工智能问答助手"},
            {"role": "user", "content": prompt},
        ]
    )
    corrected = response.choices[0].message.content
    print(f"\n[修正后的回答]{corrected}")
    state["retrieved_context"] = retrieved
    state["corrected_answer"] = corrected
    return state

# === 构建 LangGraph ===
workflow = StateGraph(state_schema=CorrectiveState)
workflow.add_node("input", input_node)
workflow.add_node("initial", initial_retrieve_and_answer)
workflow.add_node("corrective", corrective_retrieve_and_answer)
workflow.set_entry_point("input")
workflow.add_edge("input", "initial")
workflow.add_edge("initial", "corrective")
workflow.add_edge("corrective", END)
app = workflow.compile()

# === 运行 ===
if __name__ == "__main__":
    while True:
        q = input("\n请输入问题（输入 exit 退出）:\n> ")
        if q.lower() in {"exit", "quit"}:
            break
        app.invoke({"question": q})
