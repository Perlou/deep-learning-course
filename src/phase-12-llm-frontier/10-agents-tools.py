"""
Agent与工具调用
===============

学习目标：
    1. 理解LLM Agent的工作原理
    2. 掌握工具调用的实现方式
    3. 了解RAG检索增强生成
"""

import json


# ==================== 第一部分：Agent概述 ====================


def introduction():
    """Agent介绍"""
    print("=" * 60)
    print("第一部分：LLM Agent概述")
    print("=" * 60)

    print("""
什么是LLM Agent？

Agent = LLM + 工具 + 记忆 + 规划

核心能力：
    1. 理解用户意图
    2. 规划执行步骤
    3. 调用外部工具
    4. 整合结果返回

ReAct模式 (Reason + Act):
┌─────────────────────────────────────────────────────────┐
│  用户: 今天北京天气怎么样？                               │
│                                                          │
│  Thought: 需要查询天气信息                                │
│  Action: weather_api(city="北京")                        │
│  Observation: 晴天，气温15-25度                          │
│  Thought: 已获取信息，可以回答                            │
│  Answer: 今天北京天气晴朗，气温15-25度                    │
└─────────────────────────────────────────────────────────┘
    """)


# ==================== 第二部分：工具调用 ====================


def tool_calling():
    """工具调用"""
    print("\n" + "=" * 60)
    print("第二部分：工具调用")
    print("=" * 60)

    print("""
工具定义格式 (OpenAI Function Calling):

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
}]

模型返回工具调用：
{
    "tool_calls": [{
        "function": {
            "name": "get_weather",
            "arguments": "{\\"city\\": \\"北京\\"}"
        }
    }]
}
    """)

    # 模拟工具调用
    print("\n模拟工具调用:")

    def get_weather(city):
        data = {"北京": "晴天 15-25度", "上海": "多云 18-22度"}
        return data.get(city, "未知城市")

    def search(query):
        return f"搜索结果: 关于'{query}'的信息..."

    tools = {"get_weather": get_weather, "search": search}

    # 模拟LLM返回
    tool_call = {"name": "get_weather", "arguments": {"city": "北京"}}

    result = tools[tool_call["name"]](**tool_call["arguments"])
    print(f"调用: {tool_call['name']}({tool_call['arguments']})")
    print(f"结果: {result}")


# ==================== 第三部分：RAG ====================


def rag_explained():
    """RAG原理"""
    print("\n" + "=" * 60)
    print("第三部分：RAG (检索增强生成)")
    print("=" * 60)

    print("""
RAG: Retrieval-Augmented Generation

解决LLM的问题：
    - 幻觉：编造不存在的信息
    - 时效性：训练数据截止
    - 领域知识：缺乏专业知识

RAG流程：
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  用户问题 → [Embedding] → 向量检索 → 相关文档             │
│                                    ↓                     │
│              LLM  ←───────── 增强Prompt                  │
│               ↓                                          │
│             回答 (带引用)                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘

Prompt模板：
    基于以下参考信息回答问题：
    
    参考信息：
    {retrieved_documents}
    
    问题：{user_question}
    
优势：
    - 减少幻觉（有据可查）
    - 知识可更新（更新文档库）
    - 可追溯来源
    """)


# ==================== 第四部分：实现示例 ====================


def implementation():
    """实现示例"""
    print("\n" + "=" * 60)
    print("第四部分：实现示例")
    print("=" * 60)

    print("""
1. LangChain Agent:

from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(name="search", func=search_func, description="搜索"),
    Tool(name="calculator", func=calc_func, description="计算")
]

# 创建Agent
llm = ChatOpenAI(model="gpt-4")
agent = create_openai_functions_agent(llm, tools, prompt)

# 执行
agent.invoke({"input": "计算123*456"})

2. 简单RAG:

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 建立向量库
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 检索
retriever = vectorstore.as_retriever(k=3)
docs = retriever.get_relevant_documents("问题")

# 生成
context = "\\n".join([d.page_content for d in docs])
prompt = f"基于以下信息回答：\\n{context}\\n问题：{question}"
answer = llm(prompt)
    """)


def main():
    introduction()
    tool_calling()
    rag_explained()
    implementation()

    print("\n" + "=" * 60)
    print("Phase 12 全部课程完成！")
    print("=" * 60)
    print("""
恭喜完成LLM前沿技术学习！

课程回顾：
    01. LLM架构 (RMSNorm, SwiGLU, GQA)
    02. 分词器 (BPE, SentencePiece)
    03. Flash Attention
    04. 预训练基础
    05. 指令微调与LoRA
    06. RLHF与DPO
    07. 模型量化
    08. 推理优化
    09. 多模态模型
    10. Agent与RAG

下一步建议：
    1. 微调一个开源模型
    2. 部署推理服务
    3. 构建RAG应用
    4. 探索Agent开发
    """)


if __name__ == "__main__":
    main()
