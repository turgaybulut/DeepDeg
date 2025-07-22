from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model: nn.Module, config: Any, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.MSELoss()
        
        self.model_save_path = Path(config.output.results_dir) / config.output.model_artifacts.model_filename
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            verbose=True,
            path=self.model_save_path,
            config=self.config
        )

    def _create_optimizer(self) -> optim.Optimizer:
        lr = self.config.training.learning_rate
        if self.config.training.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr)
        else:
            return optim.Adam(self.model.parameters(), lr=lr)

    def _create_scheduler(self) -> optim.lr_scheduler.ReduceLROnPlateau:
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.training.reduce_lr_factor,
            patience=self.config.training.reduce_lr_patience,
            min_lr=self.config.training.min_lr
        )

    def train(self) -> None:
        for epoch in range(self.config.training.epochs):
            self.model.train()
            train_loss = 0.0
            for emb, feat, target in self.train_loader:
                emb, feat, target = emb.to(self.device), feat.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(emb, feat)
                loss = self.criterion(output, target.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = self.evaluate()

            print(f"Epoch {epoch+1}/{self.config.training.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            self.scheduler.step(avg_val_loss)
            self.early_stopping(avg_val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        
        artifacts = torch.load(self.early_stopping.path, weights_only=False)
        self.model.load_state_dict(artifacts['model_state_dict'])
        self.save_artifacts()

    def evaluate(self) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb, feat, target in self.val_loader:
                emb, feat, target = emb.to(self.device), feat.to(self.device), target.to(self.device)
                output = self.model(emb, feat)
                loss = self.criterion(output, target.unsqueeze(1))
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def save_artifacts(self) -> None:
        artifacts = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(artifacts, self.model_save_path)
        print(f"Model and artifacts saved to {self.model_save_path}")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=Path('checkpoint.pt'), config=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        artifacts = {
            'model_state_dict': model.state_dict(),
            'config': self.config
        }
        torch.save(artifacts, self.path)
        self.val_loss_min = val_loss