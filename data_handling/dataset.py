from typing import Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

class DeepDegDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, features: np.ndarray, targets: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings.copy()).float()
        self.features = torch.from_numpy(features.copy()).float()
        self.targets = torch.from_numpy(targets.copy()).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.features[idx], self.targets[idx]

def create_dataloaders(
    config: Any,
    embeddings_split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    features_split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    targets_split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    test_only: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    X_train_emb, X_val_emb, X_test_emb = embeddings_split
    X_train_feat, X_val_feat, X_test_feat = features_split
    y_train, y_val, y_test = targets_split

    if test_only:
        if X_test_emb is None or X_test_feat is None or y_test is None:
            raise ValueError("Test data must be provided when test_only is True.")
        test_dataset = DeepDegDataset(X_test_emb, X_test_feat, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
        return None, None, test_loader

    train_loader, val_loader, test_loader = None, None, None

    if X_train_emb is not None and X_train_feat is not None and y_train is not None:
        train_dataset = DeepDegDataset(X_train_emb, X_train_feat, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)

    if X_val_emb is not None and X_val_feat is not None and y_val is not None:
        val_dataset = DeepDegDataset(X_val_emb, X_val_feat, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    if X_test_emb is not None and X_test_feat is not None and y_test is not None:
        test_dataset = DeepDegDataset(X_test_emb, X_test_feat, y_test)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def split_data(
    config: Any,
    embeddings: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X_train_val_emb, X_test_emb, X_train_val_feat, X_test_feat, y_train_val, y_test = train_test_split(
        embeddings, features, targets, test_size=config.data.split.test_size, random_state=config.reproducibility.random_seed
    )

    X_train_emb, X_val_emb, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
        X_train_val_emb, X_train_val_feat, y_train_val, test_size=config.data.split.val_size, random_state=config.reproducibility.random_seed
    )
    
    return X_train_emb, X_val_emb, X_test_emb, X_train_feat, X_val_feat, X_test_feat, y_train, y_val, y_test