from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.config import get_config
from data_handling.dataset import create_dataloaders, split_data
from data_handling.feature_processor import FeatureProcessor
from models.deepdeg import DeepDeg
from training.evaluator import Evaluator
from training.shap_analyzer import ShapAnalyzer

def main():
    # --- 1. Configuration and Setup ---
    local_config = get_config(config_path='config.yaml')
    results_dir = Path(local_config.output.results_dir)
    model_path = results_dir / local_config.output.model_artifacts.model_filename

    artifacts = torch.load(model_path, weights_only=False)
    config = artifacts['config']

    # --- 2. Data Loading ---
    embeddings = np.load(config.data.paths.merged_embeddings)
    feature_df = pd.read_csv(config.data.paths.feature_data)
    targets = feature_df[config.data.columns.target].values
    feature_names = [col for col in feature_df.columns if col not in [config.data.columns.feature_id, config.data.columns.target]]
    features = feature_df[feature_names].values

    if len(embeddings.shape) == 2:
        embeddings = embeddings.reshape((*embeddings.shape, 1))

    # --- 3. Data Splitting ---
    (X_train_emb, X_val_emb, X_test_emb, 
     X_train_feat, X_val_feat, X_test_feat, 
     y_train, y_val, y_test) = split_data(config, embeddings, features, targets)

    # --- 4. Feature Processing ---
    feature_processor = FeatureProcessor(config)
    feature_processor.load(str(results_dir))

    _, _, X_test_norm = feature_processor.normalize_features(X_train_feat, X_val_feat, X_test_feat)
    
    selected_indices, selected_feature_names = feature_processor.get_selected_features(feature_names)
    X_test_selected = X_test_norm[:, selected_indices]

    # --- 5. Create DataLoader ---
    _, _, test_loader = create_dataloaders(
        config,
        (X_train_emb, X_val_emb, X_test_emb),
        (X_train_feat, X_val_feat, X_test_selected),
        (y_train, y_val, y_test),
        test_only=True
    )

    # --- 6. Model Initialization ---
    model = DeepDeg(config, feature_input_dim=X_test_selected.shape[1])
    model.load_state_dict(artifacts['model_state_dict'])
    print(f"Loaded model from {model_path}")

    # --- 7. Evaluation ---
    evaluator = Evaluator(model, config, test_loader)
    metrics = evaluator.evaluate()

    print("\n--- Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"{key.upper():<10}: {value:.4f}")
    print("--------------------------")

    # --- 8. Save Results ---
    metrics_df = pd.DataFrame([metrics])
    metrics_path = results_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved evaluation metrics to {metrics_path}")

    # --- 9. SHAP Analysis ---
    if config.shap.enabled:
        shap_analyzer = ShapAnalyzer(model, config, test_loader, selected_feature_names)
        shap_analyzer.analyze()

if __name__ == "__main__":
    main()
