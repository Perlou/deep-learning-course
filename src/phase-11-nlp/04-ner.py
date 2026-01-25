"""
命名实体识别 (NER)
==================

学习目标：
    1. 理解命名实体识别任务
    2. 掌握 BIO 标注方案
    3. 实现 BiLSTM-CRF 模型
    4. 使用预训练模型进行 NER

核心概念：
    - NER：识别文本中的实体及其类型
    - BIO 标注：B-开始，I-内部，O-非实体
    - CRF：条件随机场，建模标签转移
    - 序列标注：为每个 token 分配标签

前置知识：
    - Phase 6: RNN/LSTM
    - 03-text-classification.py
"""

import torch
import torch.nn as nn
import numpy as np


# ==================== 第一部分：NER 任务概述 ====================


def introduction():
    """NER 任务介绍"""
    print("=" * 60)
    print("第一部分：命名实体识别概述")
    print("=" * 60)

    print("""
命名实体识别 (Named Entity Recognition, NER)

任务定义：
    识别文本中的实体（人名、地名、机构名等）并标注类型

示例：
    输入: "乔布斯 于 2011年 在 加州 创立了 苹果公司"
    
    输出:
    ┌────────┬──────────┬─────────┐
    │  实体   │   类型   │   标签  │
    ├────────┼──────────┼─────────┤
    │ 乔布斯  │ 人名     │ PER     │
    │ 2011年 │ 时间     │ TIME    │
    │ 加州    │ 地名     │ LOC     │
    │ 苹果公司│ 机构名   │ ORG     │
    └────────┴──────────┴─────────┘

常见实体类型：
    - PER (Person): 人名
    - LOC (Location): 地名
    - ORG (Organization): 机构名
    - TIME: 时间
    - MISC: 其他
    """)


# ==================== 第二部分：BIO 标注方案 ====================


def bio_tagging():
    """BIO 标注方案"""
    print("\n" + "=" * 60)
    print("第二部分：BIO 标注方案")
    print("=" * 60)

    print("""
BIO 标注方案：
    B-XXX: 实体 XXX 的开始 (Beginning)
    I-XXX: 实体 XXX 的内部 (Inside)
    O: 非实体 (Outside)

示例：
    词语:  乔   布   斯   于   2011  年   在   加   州
    标签: B-PER I-PER I-PER O  B-TIME I-TIME O  B-LOC I-LOC

BIOES 方案（更精细）：
    B: Beginning
    I: Inside
    O: Outside
    E: End (实体结束)
    S: Single (单字实体)

标签转移约束：
    - I-XXX 只能跟在 B-XXX 或 I-XXX 后面
    - 不能: O → I-PER (错误！)
    - 可以: B-PER → I-PER (正确)
    """)


# ==================== 第三部分：BiLSTM-CRF 模型 ====================


class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF 命名实体识别模型"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags):
        super().__init__()

        self.num_tags = num_tags
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # 发射分数层
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CRF 转移矩阵
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # 特殊标签索引
        self.start_tag = num_tags - 2
        self.end_tag = num_tags - 1

    def _get_lstm_features(self, x):
        """获取 LSTM 输出的发射分数"""
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def forward(self, x, tags):
        """
        训练时计算负对数似然损失
        """
        emissions = self._get_lstm_features(x)

        # 计算配分函数（所有路径的分数和）
        forward_score = self._forward_algorithm(emissions)

        # 计算正确路径的分数
        gold_score = self._score_sentence(emissions, tags)

        # 损失 = log(Σexp(all_paths)) - score(gold_path)
        loss = forward_score - gold_score
        return loss.mean()

    def _forward_algorithm(self, emissions):
        """前向算法计算配分函数"""
        batch_size, seq_len, num_tags = emissions.shape

        # 初始化
        alpha = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        alpha[:, self.start_tag] = 0

        for t in range(seq_len):
            emit_score = emissions[:, t, :].unsqueeze(1)  # (batch, 1, num_tags)
            trans_score = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            alpha_expand = alpha.unsqueeze(2)  # (batch, num_tags, 1)

            # (batch, num_tags, num_tags)
            score = alpha_expand + trans_score + emit_score
            alpha = torch.logsumexp(score, dim=1)

        # 加上结束标签的转移
        alpha = alpha + self.transitions[self.end_tag, :]
        return torch.logsumexp(alpha, dim=1)

    def _score_sentence(self, emissions, tags):
        """计算给定标签序列的分数"""
        batch_size, seq_len, _ = emissions.shape

        score = torch.zeros(batch_size, device=emissions.device)

        # 转移分数 + 发射分数
        prev_tags = torch.full(
            (batch_size,), self.start_tag, dtype=torch.long, device=emissions.device
        )

        for t in range(seq_len):
            emit_score = emissions[torch.arange(batch_size), t, tags[:, t]]
            trans_score = self.transitions[tags[:, t], prev_tags]
            score = score + emit_score + trans_score
            prev_tags = tags[:, t]

        # 结束转移
        score = score + self.transitions[self.end_tag, prev_tags]
        return score

    def decode(self, x):
        """维特比解码，返回最优标签序列"""
        emissions = self._get_lstm_features(x)
        return self._viterbi_decode(emissions)

    def _viterbi_decode(self, emissions):
        """维特比算法"""
        batch_size, seq_len, num_tags = emissions.shape

        # 初始化
        viterbi = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        viterbi[:, self.start_tag] = 0
        backpointers = []

        for t in range(seq_len):
            emit_score = emissions[:, t, :].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            viterbi_expand = viterbi.unsqueeze(2)

            score = viterbi_expand + trans_score + emit_score
            viterbi, best_prev = score.max(dim=1)
            backpointers.append(best_prev)

        # 回溯
        viterbi = viterbi + self.transitions[self.end_tag, :]
        _, best_last = viterbi.max(dim=1)

        best_paths = [best_last.unsqueeze(1)]
        for bp in reversed(backpointers):
            best_prev = bp[torch.arange(batch_size), best_paths[-1].squeeze(1)]
            best_paths.append(best_prev.unsqueeze(1))

        best_paths.reverse()
        return torch.cat(best_paths[1:], dim=1)


def bilstm_crf_demo():
    """BiLSTM-CRF 演示"""
    print("\n" + "=" * 60)
    print("第三部分：BiLSTM-CRF 模型")
    print("=" * 60)

    print("""
BiLSTM-CRF 架构：
    
    输入 → Embedding → BiLSTM → Linear → CRF → 标签序列
    
    - BiLSTM: 捕捉双向上下文信息
    - CRF: 建模标签之间的转移约束
    
为什么需要 CRF？
    - 单独 LSTM 可能输出 I-PER 跟在 B-LOC 后面（不合法）
    - CRF 学习标签转移概率，避免非法序列
    """)

    # 创建模型
    vocab_size = 5000
    embed_dim = 100
    hidden_dim = 128
    num_tags = 9 + 2  # BIO tags + START + END

    model = BiLSTM_CRF(vocab_size, embed_dim, hidden_dim, num_tags)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟推理
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 解码
    best_tags = model.decode(x)
    print(f"输入形状: {x.shape}")
    print(f"预测标签形状: {best_tags.shape}")


# ==================== 第四部分：使用预训练模型 ====================


def pretrained_ner():
    """使用预训练模型进行 NER"""
    print("\n" + "=" * 60)
    print("第四部分：使用预训练模型")
    print("=" * 60)

    print("""
使用 HuggingFace 进行 NER：

    from transformers import pipeline
    
    # 加载 NER pipeline
    ner = pipeline("ner", model="bert-base-chinese")
    
    # 预测
    text = "乔布斯于2011年在加州创立了苹果公司"
    results = ner(text)
    
    for entity in results:
        print(f"{entity['word']}: {entity['entity']} ({entity['score']:.4f})")

微调 BERT for NER：

    from transformers import BertForTokenClassification, Trainer
    
    model = BertForTokenClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=num_tags
    )
    
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
练习 1：实现 BIO 标签转换
    任务：将实体列表转换为 BIO 标签序列

练习 1 答案：
    def to_bio(tokens, entities):
        '''
        entities: [(start, end, type), ...]
        '''
        tags = ['O'] * len(tokens)
        
        for start, end, ent_type in entities:
            tags[start] = f'B-{ent_type}'
            for i in range(start + 1, end):
                tags[i] = f'I-{ent_type}'
        
        return tags
    
    # 示例
    tokens = ["乔", "布", "斯", "是", "CEO"]
    entities = [(0, 3, "PER")]
    tags = to_bio(tokens, entities)
    # ['B-PER', 'I-PER', 'I-PER', 'O', 'O']

练习 2：评估 NER 模型
    任务：计算精确率、召回率、F1

练习 2 答案：
    from seqeval.metrics import f1_score, classification_report
    
    true_labels = [['B-PER', 'I-PER', 'O', 'B-LOC']]
    pred_labels = [['B-PER', 'I-PER', 'O', 'O']]
    
    f1 = f1_score(true_labels, pred_labels)
    print(classification_report(true_labels, pred_labels))

思考题 1：为什么 NER 用 CRF 而不是普通 softmax？
答案：
    - Softmax 独立预测每个位置
    - CRF 考虑标签转移约束（如 I 必须跟在 B 后）
    - CRF 生成的序列更合法

思考题 2：嵌套实体如何处理？
答案：
    - 普通 BIO 无法处理嵌套
    - 方案：多层标注、span预测、机器阅读理解方法
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    bio_tagging()
    bilstm_crf_demo()
    pretrained_ner()
    exercises()

    print("\n" + "=" * 60)
    print("课程完成！下一步：05-question-answering.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
