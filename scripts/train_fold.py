import copy
import gc
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch

from config.config import get_config
from data_handling.dataset import create_dataloaders
from data_handling.feature_processor import FeatureProcessor
from models.deepdeg import DeepDeg
from training.evaluator import Evaluator
from training.trainer import Trainer
from utils.helpers import set_seed

def main():
    # --- 1. Configuration and Setup ---
    config = get_config(config_path='config.yaml')
    set_seed(config.reproducibility.random_seed)
    
    base_results_dir = Path(config.output.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Data Loading ---
    embeddings = np.load(config.data.paths.merged_embeddings)
    feature_df = pd.read_csv(config.data.paths.feature_data)
    targets = feature_df[config.data.columns.target].values
    feature_names = [col for col in feature_df.columns if col not in [config.data.columns.feature_id, config.data.columns.target]]
    features = feature_df[feature_names].values

    if len(embeddings.shape) == 2:
        embeddings = embeddings.reshape((*embeddings.shape, 1))

    # --- 3. Cross-Validation Setup ---
    kf = KFold(n_splits=10, shuffle=True, random_state=config.reproducibility.random_seed)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(features)):
        print(f"\n--- Fold {fold+1}/10 ---")

        # --- 4. Create fold-specific results directory ---
        fold_results_dir = base_results_dir / f"fold_{fold+1:02d}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)
        
        fold_config = copy.deepcopy(config)
        fold_config.output.results_dir = str(fold_results_dir)

        # --- 5. Data Splitting for Current Fold ---
        X_train_emb, X_val_emb = embeddings[train_index], embeddings[val_index]
        X_train_feat, X_val_feat = features[train_index], features[val_index]
        y_train, y_val = targets[train_index], targets[val_index]

        # --- 6. Feature Processing ---
        feature_processor = FeatureProcessor(fold_config)
        X_train_selected, X_val_selected, _, selected_feature_names = feature_processor.process_features(
            X_train_feat, y_train, X_val_feat, None, feature_names
        )
        print(f"Selected {len(selected_feature_names)} features for fold {fold+1}.")

        # --- 7. Create DataLoaders ---
        train_loader, val_loader, _ = create_dataloaders(
            fold_config,
            (X_train_emb, X_val_emb, None),
            (X_train_selected, X_val_selected, None),
            (y_train, y_val, None)
        )

        # --- 8. Model Initialization ---
        model = DeepDeg(fold_config, feature_input_dim=X_train_selected.shape[1])
        
        # --- 9. Training ---
        trainer = Trainer(model, fold_config, train_loader, val_loader)
        trainer.train()

        # --- 10. Evaluation ---
        evaluator = Evaluator(model, fold_config, val_loader)
        metrics = evaluator.evaluate()
        fold_results.append(metrics)
        print(f"Fold {fold+1} Metrics: {metrics}")

        # --- 11. Save fold artifacts ---
        feature_processor.save(str(fold_results_dir))
        print(f"Saved fold {fold+1} artifacts to {fold_results_dir}")

        # --- 12. Cleanup for next fold ---
        del model, trainer, evaluator, train_loader, val_loader, feature_processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.mps.empty_cache() if torch.mps.is_available() else None
        gc.collect()

    # --- 13. Aggregate and Save Results ---
    avg_metrics = {metric: mean([res[metric] for res in fold_results]) for metric in fold_results[0]}
    std_metrics = {metric: stdev([res[metric] for res in fold_results]) for metric in fold_results[0]}

    print("\n--- Cross-Validation Results ---")
    for metric, avg in avg_metrics.items():
        print(f"Average {metric}: {avg:.4f} (+/- {std_metrics[metric]:.4f})")

    results_df = pd.DataFrame(fold_results)
    results_df.loc['mean'] = results_df.mean()
    results_df.loc['std'] = results_df.std()
    
    cv_results_path = base_results_dir / "cross_validation_results.csv"
    results_df.to_csv(cv_results_path)
    print(f"\nSaved cross-validation results to {cv_results_path}")

if __name__ == "__main__":
    main()

