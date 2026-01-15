"""
训练中英翻译模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from translation_model import (
    TransformerTranslator,
    create_sample_data,
    Vocabulary,
    tokenize_chinese,
    tokenize_english,
)


class TranslationDataset(Dataset):
    """翻译数据集"""

    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch, src_vocab, tgt_vocab):
    """批处理"""
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_batch.append(src_vocab.encode(src))
        tgt_batch.append(tgt_vocab.encode(tgt))

    # Padding
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)

    src_padded, tgt_padded = [], []

    for src in src_batch:
        src_padded.append(src + [0] * (src_max_len - len(src)))

    for tgt in tgt_batch:
        tgt_padded.append(tgt + [0] * (tgt_max_len - len(tgt)))

    return torch.LongTensor(src_padded), torch.LongTensor(tgt_padded)


def train_epoch(model, dataloader, criterion, optimizer, device, src_vocab, tgt_vocab):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # tgt_input去掉最后一个token，tgt_output去掉第一个token
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 创建掩码
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        src_padding_mask = (src == 0).to(device)
        tgt_padding_mask = (tgt_input == 0).to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(
            src,
            tgt_input,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # 计算损失（忽略padding）
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("=" * 60)
    print("训练中英翻译模型")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    zh_data, en_data = create_sample_data()

    # 构建词表
    zh_vocab = Vocabulary()
    en_vocab = Vocabulary()

    for sent in zh_data:
        zh_vocab.add_sentence(sent)
    for sent in en_data:
        en_vocab.add_sentence(sent)

    zh_vocab.build_vocab()
    en_vocab.build_vocab()

    print(f"中文词表大小: {len(zh_vocab)}")
    print(f"英文词表大小: {len(en_vocab)}")

    # 创建数据加载器
    dataset = TranslationDataset(zh_data, en_data)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, zh_vocab, en_vocab),
    )

    # 创建模型
    print("\n创建模型...")
    model = TransformerTranslator(
        src_vocab_size=len(zh_vocab),
        tgt_vocab_size=len(en_vocab),
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # 训练
    num_epochs = 100
    print(f"\n开始训练 {num_epochs} epochs...")
    print("-" * 60)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model, dataloader, criterion, optimizer, device, zh_vocab, en_vocab
        )

        elapsed = time.time() - start_time

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | Loss: {train_loss:.4f} | Time: {elapsed:.2f}s"
            )

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "zh_vocab": zh_vocab,
                "en_vocab": en_vocab,
            }
            torch.save(checkpoint, "outputs/checkpoints/best_model.pth")

    print("-" * 60)
    print(f"\n训练完成! 最佳损失: {best_loss:.4f}")
    print(f"模型已保存到: outputs/checkpoints/best_model.pth")

    # 测试翻译
    print("\n测试翻译:")
    model.eval()
    for i in range(3):
        src_sent = zh_data[i]
        src_encoded = zh_vocab.encode(src_sent, add_sos_eos=False)

        translation = model.translate(src_encoded, zh_vocab, en_vocab, device=device)

        print(f"\n  中文: {' '.join(src_sent)}")
        print(f"  翻译: {' '.join(translation)}")
        print(f"  参考: {' '.join(en_data[i])}")


if __name__ == "__main__":
    main()
