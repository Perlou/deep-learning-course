import numpy as np


def basic_attention(Q, K, V):
    d_k = K.shape[-1]

    scores = np.matmul(Q, K.T)

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = np.exp(scaled_scores) / np.sum(
        np.exp(scaled_scores), axis=-1, keepdims=True
    )

    output = np.matmul(attention_weights, V)

    return output, attention_weights


# 示例
np.random.seed(42)
Q = np.random.randn(1, 64)  # 1个查询
K = np.random.randn(10, 64)  # 10个键
V = np.random.randn(10, 64)  # 10个值

output, weights = basic_attention(Q, K, V)
print(f"输出形状: {output.shape}")  # (1, 64)
print(f"注意力权重: {weights.round(3)}")  # 每个键的重要程度
