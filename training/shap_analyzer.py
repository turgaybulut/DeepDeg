import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from typing import Any, List
from pathlib import Path

class ShapAnalyzer:
    def __init__(self, model: torch.nn.Module, config: Any, test_loader: DataLoader, feature_names: List[str]):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.feature_names = feature_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def analyze(self):
        print("\n--- Starting SHAP Analysis ---")
        self.model.eval()

        background_indices = np.random.choice(len(self.test_loader.dataset), self.config.shap.num_background_samples, replace=False)
        test_indices = np.random.choice(len(self.test_loader.dataset), self.config.shap.num_test_samples, replace=False)

        background_loader = DataLoader(Subset(self.test_loader.dataset, background_indices), batch_size=self.config.shap.num_background_samples)
        test_loader_subset = DataLoader(Subset(self.test_loader.dataset, test_indices), batch_size=self.config.shap.num_test_samples)

        background_emb, background_feat, _ = next(iter(background_loader))
        test_emb, test_feat, _ = next(iter(test_loader_subset))

        background_emb = background_emb.to(self.device)
        background_feat = background_feat.to(self.device)
        test_emb = test_emb.to(self.device)
        test_feat = test_feat.to(self.device)

        explainer = shap.DeepExplainer(self.model, [background_emb, background_feat])

        shap_values = explainer.shap_values([test_emb, test_feat], check_additivity=False)

        shap_values_emb = shap_values[0]
        shap_values_feat = shap_values[1]

        if len(shap_values_emb.shape) > 2:
            shap_values_emb = shap_values_emb.reshape(shap_values_emb.shape[0], -1)

        if len(shap_values_feat.shape) > 2:
            shap_values_feat = shap_values_feat.reshape(shap_values_feat.shape[0], -1)

        combined_shap_values = np.hstack((shap_values_emb, shap_values_feat))

        test_emb_numpy = test_emb.cpu().numpy()
        if len(test_emb_numpy.shape) > 2:
            test_emb_numpy = test_emb_numpy.reshape(test_emb_numpy.shape[0], -1)
        
        combined_test_data = np.hstack((test_emb_numpy, test_feat.cpu().numpy()))

        embedding_feature_names = [f"embedding_{i}" for i in range(test_emb_numpy.shape[1])]
        combined_feature_names = embedding_feature_names + self.feature_names

        self.plot_and_save(combined_shap_values, combined_test_data, combined_feature_names)
        
        self.save_shap_results_to_csv(combined_shap_values, combined_feature_names)

        print("--- SHAP Analysis Finished ---")

    def plot_and_save(self, shap_values: np.ndarray, test_data: np.ndarray, feature_names: List[str]):
        results_dir = Path(self.config.output.results_dir)
        
        if len(shap_values.shape) != 2:
            shap_values_reshaped = shap_values.reshape(shap_values.shape[0], -1)
        else:
            shap_values_reshaped = shap_values

        plt.figure()
        shap.summary_plot(
            shap_values_reshaped, 
            test_data, 
            feature_names=feature_names, 
            max_display=self.config.shap.max_display, 
            show=False
        )
        summary_path = results_dir / "shap_summary_plot.png"
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP summary plot to {summary_path}")

        plt.figure()
        shap.summary_plot(
            shap_values_reshaped, 
            feature_names=feature_names, 
            plot_type="bar",
            max_display=self.config.shap.max_display,
            show=False
        )
        bar_path = results_dir / "shap_bar_plot.png"
        plt.savefig(bar_path, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP bar plot to {bar_path}")

    def save_shap_results_to_csv(self, shap_values: np.ndarray, feature_names: List[str]):
        results_dir = Path(self.config.output.results_dir)
        output_path = results_dir / "shap_analysis_results.csv"

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_absolute_shap': mean_abs_shap
        })

        sorted_features_df = feature_importance_df.sort_values(by='mean_absolute_shap', ascending=False)

        top_100_features = sorted_features_df.head(100)
        embedding_count = top_100_features['feature'].str.startswith('embedding_').sum()
        feature_count = 100 - embedding_count
        
        embedding_ratio = embedding_count / 100.0
        feature_ratio = feature_count / 100.0

        ratio_df = pd.DataFrame({
            'type': ['embedding', 'feature'],
            'ratio': [embedding_ratio, feature_ratio]
        })

        with open(output_path, 'w') as f:
            f.write("# SHAP Analysis Top 100 Feature Ratio\n")
            ratio_df.to_csv(f, index=False)
            f.write("\n# Feature Importance (Sorted)\n")
            sorted_features_df.to_csv(f, index=False)

        print(f"Saved SHAP analysis results to {output_path}")