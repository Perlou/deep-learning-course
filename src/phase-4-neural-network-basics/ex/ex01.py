import numpy as np
import matplotlib.pyplot as plt


# 激活函数
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# 完整的神经网络类
class NeuralNetworkFromScratch:
    def __init__(self, layers):
        """
        layers: 每层神经元数量的列表
        例如: [784, 128, 64, 10] 表示输入784维，两个隐藏层，输出10类
        """
        self.layers = layers
        self.num_layers = len(layers) - 1

        # 初始化参数（He初始化）
        self.params = {}
        for i in range(self.num_layers):
            self.params[f"W{i}"] = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(
                2.0 / layers[i]
            )
            self.params[f"b{i}"] = np.zeros((1, layers[i + 1]))

        self.cache = {}

    def forward(self, X):
        """前向传播"""
        self.cache["A0"] = X
        A = X

        for i in range(self.num_layers):
            Z = np.dot(A, self.params[f"W{i}"]) + self.params[f"b{i}"]
            self.cache[f"Z{i}"] = Z

            if i == self.num_layers - 1:
                A = softmax(Z)  # 输出层
            else:
                A = relu(Z)  # 隐藏层

            self.cache[f"A{i + 1}"] = A

        return A

    def backward(self, Y):
        """反向传播"""
        m = Y.shape[0]
        grads = {}

        # 输出层梯度
        dZ = self.cache[f"A{self.num_layers}"] - Y

        for i in range(self.num_layers - 1, -1, -1):
            A_prev = self.cache[f"A{i}"]

            grads[f"W{i}"] = np.dot(A_prev.T, dZ) / m
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA = np.dot(dZ, self.params[f"W{i}"].T)
                dZ = dA * relu_derivative(self.cache[f"Z{i - 1}"])

        return grads

    def update_params(self, grads, learning_rate):
        """更新参数"""
        for i in range(self.num_layers):
            self.params[f"W{i}"] -= learning_rate * grads[f"W{i}"]
            self.params[f"b{i}"] -= learning_rate * grads[f"b{i}"]

    def train(
        self, X_train, Y_train, X_val, Y_val, epochs=1000, lr=0.01, batch_size=32
    ):
        """训练模型"""
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            # 小批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]

                # 前向传播
                self.forward(X_batch)

                # 反向传播
                grads = self.backward(Y_batch)

                # 更新参数
                self.update_params(grads, lr)

            # 记录指标
            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)

            train_loss = cross_entropy(Y_train, train_pred)
            val_loss = cross_entropy(Y_val, val_pred)
            train_acc = np.mean(
                np.argmax(train_pred, axis=1) == np.argmax(Y_train, axis=1)
            )
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(Y_val, axis=1))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}"
                )

        return history

    def predict(self, X):
        """预测"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 生成示例数据（多分类）
    np.random.seed(42)

    # 创建螺旋数据集
    def create_spiral_data(samples_per_class, classes):
        X = np.zeros((samples_per_class * classes, 2))
        Y = np.zeros((samples_per_class * classes, classes))

        for class_idx in range(classes):
            idx = range(
                samples_per_class * class_idx, samples_per_class * (class_idx + 1)
            )
            r = np.linspace(0.0, 1, samples_per_class)
            t = (
                np.linspace(class_idx * 4, (class_idx + 1) * 4, samples_per_class)
                + np.random.randn(samples_per_class) * 0.2
            )

            X[idx] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
            Y[idx, class_idx] = 1

        return X, Y

    # 生成数据
    X, Y = create_spiral_data(100, 3)

    # 分割数据
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    # 创建并训练模型
    model = NeuralNetworkFromScratch([2, 64, 32, 3])
    history = model.train(
        X_train, Y_train, X_val, Y_val, epochs=1000, lr=0.1, batch_size=32
    )

    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss Curve")

    # 准确率曲线
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy Curve")

    # 决策边界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[2].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    axes[2].scatter(
        X[:, 0], X[:, 1], c=np.argmax(Y, axis=1), cmap=plt.cm.RdYlBu, edgecolors="black"
    )
    axes[2].set_title("Decision Boundary")

    plt.tight_layout()
    plt.show()
