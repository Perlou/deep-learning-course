"""
callbacks.py — 自定义训练回调 (HF Trainer 版)
=============================================

对应 from-scratch 版的 TrainingLogger + 手动打印逻辑。
HF Trainer 通过 Callback 机制提供相同的训练过程可视化。

from-scratch 对比:
  - TrainingLogger.log() → on_log()
  - PreTrainer.__init__ 打印 → on_train_begin()
  - _validate_loss() 打印 → on_evaluate()
  - _finalize() 打印 → on_train_end()
"""

from transformers import TrainerCallback


class ClearMindLoggingCallback(TrainerCallback):
    """ClearMind 训练日志回调

    提供简洁、格式化的训练过程输出，
    替代 from-scratch 版 TrainingLogger 的手动打印。
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时打印配置摘要"""
        model = kwargs.get("model")
        print("\n" + "=" * 60)
        print("ClearMind 训练开始")
        print("=" * 60)
        if model is not None:
            num_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  模型参数: {num_params:,} (可训练: {trainable:,})")
        print(f"  最大步数: {args.max_steps}")
        print(f"  批次大小: {args.per_device_train_batch_size}")
        print(f"  梯度累积: {args.gradient_accumulation_steps}")
        print(f"  有效批次: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  学习率: {args.learning_rate}")
        print(f"  调度器: {args.lr_scheduler_type}")
        print(f"  预热步数: {args.warmup_steps}")
        print(f"  设备: {args.device}")
        print("=" * 60 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """格式化输出训练指标"""
        if logs is None:
            return
        step = state.global_step
        parts = [f"step {step:>6d}"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        print(" | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """打印验证结果"""
        if metrics is None:
            return
        print(f"\n--- 验证 (step {state.global_step}) ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

    def on_train_end(self, args, state, control, **kwargs):
        """打印训练总结"""
        print("\n" + "=" * 60)
        print("ClearMind 训练完成")
        print("=" * 60)
        print(f"  总步数: {state.global_step}")
        if state.best_metric is not None:
            print(f"  最佳指标: {state.best_metric:.4f}")
        print("=" * 60 + "\n")
