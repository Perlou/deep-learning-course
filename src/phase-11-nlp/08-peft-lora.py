"""
PEFT 与 LoRA
============

学习目标：
    1. 理解参数高效微调 (PEFT) 的动机
    2. 掌握 LoRA 的原理和实现
    3. 学习使用 HuggingFace PEFT 库
    4. 了解其他 PEFT 方法

核心概念：
    - PEFT：只训练少量参数的微调方法
    - LoRA：低秩适配器，分解权重矩阵
    - Adapter：在层间插入小网络
    - Prompt Tuning：只训练提示向量

前置知识：
    - 07-transformer-finetuning.py
    - 线性代数：矩阵分解
"""

import torch
import torch.nn as nn
import math


# ==================== 第一部分：PEFT 动机 ====================


def introduction():
    """PEFT 动机介绍"""
    print("=" * 60)
    print("第一部分：PEFT 动机")
    print("=" * 60)

    print("""
为什么需要参数高效微调 (PEFT)？

全量微调的问题：
┌─────────────────────────────────────────────────────────────┐
│  模型大小       参数量          全量微调显存需求              │
├─────────────────────────────────────────────────────────────┤
│  BERT-base      110M           ~1GB                         │
│  BERT-large     340M           ~4GB                         │
│  GPT-2          1.5B           ~12GB                        │
│  LLaMA-7B       7B             ~56GB                        │
│  LLaMA-70B      70B            ~560GB                       │
└─────────────────────────────────────────────────────────────┘

PEFT 的优势：
    1. 减少显存：只更新 0.1%~1% 的参数
    2. 快速训练：参数少，计算量小
    3. 多任务：每个任务只需存储小型适配器
    4. 防止遗忘：预训练参数保持不变

常见 PEFT 方法：
    - LoRA (Low-Rank Adaptation)
    - Adapter
    - Prefix-Tuning
    - Prompt Tuning
    - P-Tuning
    """)


# ==================== 第二部分：LoRA 原理 ====================


def lora_theory():
    """LoRA 原理详解"""
    print("\n" + "=" * 60)
    print("第二部分：LoRA 原理")
    print("=" * 60)

    print("""
LoRA (Low-Rank Adaptation) 核心思想：

    预训练权重 W ∈ R^(d×k) 在微调时的更新 ΔW 是低秩的
    
    不更新 W，而是添加低秩分解：
    
    W' = W + ΔW = W + B × A
    
    其中：
    - W: 原始权重 (d × k)，冻结
    - A: 降维矩阵 (r × k)
    - B: 升维矩阵 (d × r)
    - r: 秩，r << min(d, k)

参数量对比：
    原始：d × k 个参数
    LoRA：(d × r) + (r × k) = r × (d + k) 个参数
    
    例如：d=4096, k=4096, r=8
    原始：16M 参数
    LoRA：64K 参数 (减少 250 倍)

可视化：
    
    输入 x ─→ [W (冻结)] ─→ 输出1 ─┐
      │                            ├─→ 相加 ─→ 最终输出
      └──→ [A] ─→ [B] ─→ 输出2 ────┘
    
    h = Wx + BAx = (W + BA)x
    """)


# ==================== 第三部分：LoRA 实现 ====================


class LoRALinear(nn.Module):
    """LoRA 线性层实现"""

    def __init__(self, original_linear, r=8, alpha=16):
        """
        Args:
            original_linear: 原始线性层 (冻结)
            r: LoRA 秩
            alpha: 缩放因子
        """
        super().__init__()

        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 冻结原始权重
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始输出
        h = self.original(x)

        # LoRA 增量：h + (x @ A^T) @ B^T * scaling
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T

        return h + lora_output * self.scaling


def lora_demo():
    """LoRA 演示"""
    print("\n" + "=" * 60)
    print("第三部分：LoRA 实现")
    print("=" * 60)

    # 创建原始线性层
    d_model = 768
    original = nn.Linear(d_model, d_model)
    original_params = sum(p.numel() for p in original.parameters())

    # 添加 LoRA
    lora_layer = LoRALinear(original, r=8, alpha=16)

    # 统计可训练参数
    trainable_params = sum(
        p.numel() for p in lora_layer.parameters() if p.requires_grad
    )

    print(f"原始层参数: {original_params:,}")
    print(f"LoRA 可训练参数: {trainable_params:,}")
    print(f"参数减少比例: {original_params / trainable_params:.1f}x")

    # 测试前向传播
    x = torch.randn(2, 10, d_model)
    output = lora_layer(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")


# ==================== 第四部分：使用 PEFT 库 ====================


def peft_library():
    """使用 HuggingFace PEFT 库"""
    print("\n" + "=" * 60)
    print("第四部分：使用 PEFT 库")
    print("=" * 60)

    print("""
安装：
    pip install peft

基本使用：

    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM
    
    # 1. 加载基础模型
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # 2. 配置 LoRA
    lora_config = LoraConfig(
        r=8,                      # LoRA 秩
        lora_alpha=16,            # 缩放因子
        target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 3. 创建 PEFT 模型
    peft_model = get_peft_model(model, lora_config)
    
    # 4. 查看可训练参数
    peft_model.print_trainable_parameters()
    # 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable: 0.06%
    
    # 5. 正常训练
    trainer = Trainer(model=peft_model, ...)
    trainer.train()
    
    # 6. 保存 LoRA 权重（只保存适配器）
    peft_model.save_pretrained("./lora_weights")
    
    # 7. 加载 LoRA 权重
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = PeftModel.from_pretrained(base_model, "./lora_weights")

合并 LoRA 到基础模型：

    # 将 LoRA 权重合并到原始权重
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("./merged_model")
    """)


# ==================== 第五部分：其他 PEFT 方法 ====================


def other_peft_methods():
    """其他 PEFT 方法"""
    print("\n" + "=" * 60)
    print("第五部分：其他 PEFT 方法")
    print("=" * 60)

    print("""
1. Adapter (Houlsby et al., 2019):
    - 在 Transformer 层之间插入小型网络
    - 结构：降维 → 非线性 → 升维 → 残差
    
    输入 x ─→ [Transformer Layer] ─→ h ─→ [Adapter] ─→ 输出
                                      │                  │
                                      └──── 残差连接 ────┘

2. Prefix-Tuning (Li & Liang, 2021):
    - 在每层添加可学习的前缀向量
    - 不修改模型参数，只训练前缀
    
    [Prefix] + [Input Tokens] → Transformer → Output

3. Prompt Tuning (Lester et al., 2021):
    - 只在输入层添加可学习的软提示
    - 比 Prefix-Tuning 更简单
    
    [Soft Prompt] + [Input] → Frozen Model → Output

4. QLoRA (Dettmers et al., 2023):
    - 4-bit 量化基础模型 + LoRA
    - 在单张 GPU 上微调 65B 模型
    
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

方法对比：
    | 方法          | 可训练参数 | 推理开销 | 灵活性 |
    |---------------|-----------|----------|--------|
    | Full FT       | 100%      | 无       | 最高   |
    | LoRA          | 0.1-1%    | 可合并   | 高     |
    | Adapter       | 1-5%      | 有       | 中     |
    | Prefix-Tuning | <0.1%     | 有       | 中     |
    | Prompt Tuning | <0.01%    | 有       | 低     |
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：实现简化版 LoRA
    任务：为 nn.Linear 添加 LoRA

练习 1 答案：
    class SimpleLoRA(nn.Module):
        def __init__(self, in_dim, out_dim, r=4):
            super().__init__()
            self.main = nn.Linear(in_dim, out_dim)
            self.main.weight.requires_grad = False
            
            self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        
        def forward(self, x):
            main_out = self.main(x)
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            return main_out + lora_out

练习 2：使用 PEFT 微调 GPT-2
    任务：用 LoRA 微调 GPT-2 生成诗歌

练习 2 答案：
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2LMHeadModel, Trainer
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    
    # 训练...

思考题 1：LoRA 为什么假设 ΔW 是低秩的？
答案：
    - 研究表明微调时权重变化具有低内在秩
    - 大模型过参数化，任务适配不需要更新所有维度
    - 低秩约束起到正则化作用

思考题 2：如何选择 LoRA 的目标模块？
答案：
    - Attention 的 Q/V 投影效果最好
    - 也可以加 K 投影和 FFN
    - 更多模块 = 更强表达力，但参数更多
    
    常用配置：
    - 轻量：只 Q/V
    - 标准：Q/K/V + output
    - 完整：Q/K/V + FFN
    """)


# ==================== 主函数 ====================


def main():
    """主函数"""
    introduction()
    lora_theory()
    lora_demo()
    peft_library()
    other_peft_methods()
    exercises()

    print("\n" + "=" * 60)
    print("Phase 11 NLP 课程全部完成！")
    print("=" * 60)
    print("""
Phase 11 知识回顾：
    1. 词向量：Word2Vec, GloVe, FastText
    2. 文本分类：TextCNN, LSTM
    3. 序列标注：NER, BiLSTM-CRF
    4. 阅读理解：抽取式问答
    5. HuggingFace：Transformers 生态
    6. 微调：预训练-微调范式
    7. PEFT：LoRA, Adapter, Prompt Tuning

恭喜你完成了 NLP 基础课程！
    """)


if __name__ == "__main__":
    main()
