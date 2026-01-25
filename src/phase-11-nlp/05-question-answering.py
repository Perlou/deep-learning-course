"""
问答系统 (Question Answering)
============================

学习目标：
    1. 理解问答系统类型（抽取式 vs 生成式）
    2. 掌握阅读理解任务原理
    3. 了解 BERT 问答模型架构
    4. 使用预训练模型进行问答

核心概念：
    - 抽取式 QA：从文档中抽取答案片段
    - 生成式 QA：生成答案文本
    - SQuAD：阅读理解数据集
    - Span Prediction：预测答案起止位置

前置知识：
    - Phase 7: Transformer
    - 04-ner.py: 序列标注
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 第一部分：问答系统概述 ====================


def introduction():
    """问答系统介绍"""
    print("=" * 60)
    print("第一部分：问答系统概述")
    print("=" * 60)

    print("""
问答系统类型：

┌─────────────────────────────────────────────────────────────┐
│                    问答系统分类                              │
├──────────────────┬──────────────────────────────────────────┤
│   抽取式 QA      │   从给定文档中抽取答案片段                 │
│                  │   Q: 谁创立了苹果？                        │
│                  │   文档: 乔布斯创立了苹果公司...            │
│                  │   A: 乔布斯 (从文档中抽取)                 │
├──────────────────┼──────────────────────────────────────────┤
│   生成式 QA      │   生成答案文本                            │
│                  │   Q: 什么是深度学习？                      │
│                  │   A: 深度学习是机器学习的一个分支...       │
├──────────────────┼──────────────────────────────────────────┤
│   知识库 QA      │   从知识图谱查询答案                       │
│                  │   Q: 北京的人口是多少？                    │
│                  │   KG: (北京, 人口, 2154万)                 │
└──────────────────┴──────────────────────────────────────────┘

本课重点：抽取式问答（阅读理解）
    """)


# ==================== 第二部分：阅读理解任务 ====================


def reading_comprehension():
    """阅读理解任务"""
    print("\n" + "=" * 60)
    print("第二部分：阅读理解任务")
    print("=" * 60)

    print("""
SQuAD 数据集 (Stanford Question Answering Dataset):

    任务：给定问题和段落，找出答案在段落中的位置
    
    示例：
    ┌─────────────────────────────────────────────────────────┐
    │ 段落 (Context):                                          │
    │ "深度学习是机器学习的一个分支，它使用多层神经网络来学习   │
    │  数据的层次化表示。深度学习在图像识别、语音识别等领域     │
    │  取得了巨大成功。"                                       │
    │                                                          │
    │ 问题 (Question):                                         │
    │ "深度学习使用什么来学习数据表示？"                        │
    │                                                          │
    │ 答案 (Answer):                                           │
    │ "多层神经网络"                                           │
    │ 起始位置: 21, 结束位置: 27                               │
    └─────────────────────────────────────────────────────────┘

模型任务：预测答案的起始位置和结束位置
    输入: [CLS] Question [SEP] Context [SEP]
    输出: start_position, end_position
    """)


# ==================== 第三部分：BERT 问答模型 ====================


class BertForQA(nn.Module):
    """BERT 问答模型（简化版）"""

    def __init__(self, hidden_size=768):
        super().__init__()
        # 在实际中，这里应该加载预训练 BERT
        # 这里用简单的线性层演示
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 预测 start 和 end

    def forward(self, sequence_output):
        """
        Args:
            sequence_output: BERT 输出 (batch, seq_len, hidden_size)

        Returns:
            start_logits: (batch, seq_len)
            end_logits: (batch, seq_len)
        """
        # 预测起止位置
        logits = self.qa_outputs(sequence_output)  # (batch, seq_len, 2)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        return start_logits, end_logits


def bert_qa_demo():
    """BERT 问答演示"""
    print("\n" + "=" * 60)
    print("第三部分：BERT 问答模型")
    print("=" * 60)

    print("""
BERT for Question Answering 架构：

    输入: [CLS] Question [SEP] Context [SEP]
          ↓
        BERT Encoder
          ↓
    sequence_output: (batch, seq_len, 768)
          ↓
    Linear(768, 2)
          ↓
    start_logits, end_logits: (batch, seq_len)
          ↓
    取 argmax 得到 start_pos, end_pos
          ↓
    answer = context[start_pos:end_pos+1]

损失函数：
    loss = CE(start_logits, start_pos) + CE(end_logits, end_pos)
    """)

    # 演示
    batch_size = 2
    seq_len = 128
    hidden_size = 768

    # 模拟 BERT 输出
    sequence_output = torch.randn(batch_size, seq_len, hidden_size)

    model = BertForQA(hidden_size)
    start_logits, end_logits = model(sequence_output)

    print(f"序列输出形状: {sequence_output.shape}")
    print(f"起始位置 logits 形状: {start_logits.shape}")
    print(f"结束位置 logits 形状: {end_logits.shape}")

    # 预测位置
    start_pos = start_logits.argmax(dim=1)
    end_pos = end_logits.argmax(dim=1)
    print(f"预测起始位置: {start_pos}")
    print(f"预测结束位置: {end_pos}")


# ==================== 第四部分：使用 HuggingFace ====================


def huggingface_qa():
    """使用 HuggingFace 进行问答"""
    print("\n" + "=" * 60)
    print("第四部分：使用 HuggingFace")
    print("=" * 60)

    print("""
使用 Pipeline 快速问答：

    from transformers import pipeline
    
    qa = pipeline("question-answering", model="bert-base-chinese")
    
    result = qa(
        question="深度学习使用什么？",
        context="深度学习使用多层神经网络来学习数据表示。"
    )
    
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['score']:.4f}")
    print(f"位置: {result['start']}-{result['end']}")

微调 BERT for QA：

    from transformers import BertForQuestionAnswering, Trainer
    
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    """)


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：实现答案抽取
    任务：给定 start_logits 和 end_logits，找到最佳答案

练习 1 答案：
    def get_best_answer(start_logits, end_logits, context_tokens, max_len=30):
        '''找到最佳答案 span'''
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        
        best_score = -float('inf')
        best_start, best_end = 0, 0
        
        for start in range(len(start_probs)):
            for end in range(start, min(start + max_len, len(end_probs))):
                score = start_probs[start] * end_probs[end]
                if score > best_score:
                    best_score = score
                    best_start, best_end = start, end
        
        answer = ''.join(context_tokens[best_start:best_end+1])
        return answer, best_score

练习 2：评估 QA 模型
    任务：实现 Exact Match 和 F1 评估指标

练习 2 答案：
    def exact_match(pred, gold):
        '''完全匹配'''
        return pred.strip() == gold.strip()
    
    def compute_f1(pred, gold):
        '''Token 级 F1'''
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())
        
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return int(pred_tokens == gold_tokens)
        
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

思考题 1：如何处理无答案的情况？
答案：
    - SQuAD 2.0 包含无答案问题
    - 给 [CLS] 的位置也算一个候选答案
    - 如果 P([CLS]) > P(best_span) + threshold，则返回无答案

思考题 2：抽取式 vs 生成式 QA 的优缺点？
答案：
    抽取式：
    - 优点：答案来自原文，更可靠
    - 缺点：无法生成原文中没有的答案
    
    生成式：
    - 优点：灵活，可以整合多处信息
    - 缺点：可能产生幻觉（编造事实）
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    reading_comprehension()
    bert_qa_demo()
    huggingface_qa()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！下一步：06-huggingface-basics.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
