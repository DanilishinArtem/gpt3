import torch
from sklearn.decomposition import PCA
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PCAWrappedLinear(nn.Module):
    def __init__(self, compressed_weights: torch.Tensor, pca_matrix: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.compressed_weights = nn.Parameter(compressed_weights)  # shape: [out_features, k]
        self.pca_matrix = pca_matrix.cuda()  # shape: [k, in_features]
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        # Восстанавливаем веса: [out_features, in_features]
        restored_weights = torch.matmul(self.compressed_weights, self.pca_matrix)  # [out_features, in_features]
        return F.linear(x, restored_weights, self.bias)  # x @ W.T + b



class Compressor:
    def __init__(self, model: nn.Module, rate: float = 0.90, max_layers: int = 9999):
        self.model = model
        self.rate = rate
        self.max_layers = max_layers
        self.compress()

    def compress(self):
        mean_compression = 0.0
        total = 0

        for name, layer in self.model.named_modules():
            if any(key in name for key in ["q_proj", "k_proj", "v_proj"]) and "out_proj" not in name:
                if not isinstance(layer, nn.Linear):
                    continue
                if total >= self.max_layers:
                    break

                W = layer.weight.data.cpu().numpy()  # shape: [out_features, in_features]
                pca = PCA()
                pca.fit(W)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                k = np.argmax(cumulative_variance >= self.rate) + 1

                W_compressed = torch.tensor(pca.transform(W)[:, :k], dtype=torch.float32)  # [out_features, k]
                pca_matrix = torch.tensor(pca.components_[:k], dtype=torch.float32)        # [k, in_features]

                out_features, in_features = W.shape
                original = out_features * in_features
                compressed = out_features * k + k * in_features
                compression = 1 - (compressed / original)
                mean_compression += compression
                total += 1

                print(f"Layer: {name}, k: {k}, original: {original}, compressed: {compressed}, compression: {compression:.4f}")

                bias = layer.bias.data if layer.bias is not None else None
                new_layer = PCAWrappedLinear(W_compressed, pca_matrix, bias)

                self.replace_module(name, new_layer)
                del layer
        # torch.cuda.empty_cache()
        print(f"Average compression: {mean_compression / total:.4f}")

    def replace_module(self, full_name: str, new_module: nn.Module):
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old_module = getattr(parent, parts[-1])
        setattr(parent, parts[-1], new_module)
        del old_module