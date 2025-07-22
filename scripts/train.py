from pathlib import Path

import numpy as np
import pandas as pd

from config.config import get_config
from data_handling.dataset import create_dataloaders, split_data
from data_handling.feature_processor import FeatureProcessor
from models.deepdeg import DeepDeg
from training.trainer import Trainer
from utils.helpers import set_seed

def main():
    # --- 1. Configuration and Setup ---
    config = get_config(config_path='config.yaml')
    set_seed(config.reproducibility.random_seed)
    
    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    X_train_selected, X_val_selected, X_test_selected, selected_feature_names = feature_processor.process_features(
        X_train_feat, y_train, X_val_feat, X_test_feat, feature_names
    )
    print(f"Selected {len(selected_feature_names)} features.")

    # --- 5. Create DataLoaders ---
    train_loader, val_loader, _ = create_dataloaders(
        config, 
        (X_train_emb, X_val_emb, X_test_emb),
        (X_train_selected, X_val_selected, X_test_selected),
        (y_train, y_val, y_test)
    )

    # --- 6. Model Initialization ---
    model = DeepDeg(config, feature_input_dim=X_train_selected.shape[1])
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # --- 7. Training ---
    trainer = Trainer(model, config, train_loader, val_loader)
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- 8. Save Artifacts ---
    feature_processor.save(str(results_dir))
    print(f"Saved feature processor artifacts to {results_dir}")

if __name__ == "__main__":
    main()