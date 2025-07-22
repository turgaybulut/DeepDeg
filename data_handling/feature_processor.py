from pathlib import Path
from typing import Any, List, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class FeatureProcessor:
    def __init__(self, config: Any):
        self.config = config
        self.scaler = StandardScaler()
        self.selector = self._init_selector()

    def _init_selector(self) -> xgb.XGBRegressor:
        params = self.config.features.selection
        return xgb.XGBRegressor(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            device=params.device,
            tree_method=params.tree_method,
            random_state=self.config.reproducibility.random_seed,
            n_jobs=-1,
        )

    def process_features(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        X_test: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        
        X_train_norm, X_val_norm, X_test_norm = self.normalize_features(X_train, X_val, X_test)
        
        X_train_selected, X_val_selected, X_test_selected, selected_feature_names = self.select_features(
            X_train_norm, y_train, X_val_norm, X_test_norm, feature_names
        )
        
        return X_train_selected, X_val_selected, X_test_selected, selected_feature_names

    def normalize_features(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test) if X_test is not None else None
        return X_train_norm, X_val_norm, X_test_norm

    def select_features(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        
        self.selector.fit(X_train, y_train)
        
        importances = self.selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        k = self.config.features.selection.k
        selected_indices = indices[:k]
        
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        X_test_selected = X_test[:, selected_indices] if X_test is not None else None
        
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        return X_train_selected, X_val_selected, X_test_selected, selected_feature_names

    def save(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        joblib.dump(self.scaler, output_dir / self.config.output.model_artifacts.scaler_filename)
        self.selector.save_model(output_dir / self.config.output.model_artifacts.selector_filename)

    def load(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        self.scaler = joblib.load(model_dir / self.config.output.model_artifacts.scaler_filename)
        self.selector.load_model(model_dir / self.config.output.model_artifacts.selector_filename)

    def get_selected_features(self, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        importances = self.selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        k = self.config.features.selection.k
        selected_indices = indices[:k]
        selected_feature_names = [feature_names[i] for i in selected_indices]
        return selected_indices, selected_feature_names
