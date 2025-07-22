from typing import Any, Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, model: nn.Module, config: Any, test_loader: DataLoader):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for emb, feat, target in self.test_loader:
                emb, feat = emb.to(self.device), feat.to(self.device)
                output = self.model(emb, feat)
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.numpy())

        preds = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()

        return self.calculate_metrics(targets, preds)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "pearson_r": pearsonr(y_true, y_pred)[0],
            "spearman_r": spearmanr(y_true, y_pred)[0],
        }
        return metrics
