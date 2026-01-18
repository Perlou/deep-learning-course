import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined_size = input_size + hidden_size
        scale = np.sqrt(2.0 / combined_size)

        self.W = np.random.randn(4 * hidden_size, combined_size) * scale
        self.b = np.zeros(4 * hidden_size)
