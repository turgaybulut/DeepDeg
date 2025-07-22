from typing import List

from pydantic import BaseModel

class ReproducibilityConfig(BaseModel):
    random_seed: int

class DataPathsConfig(BaseModel):
    sequence_data: str
    feature_data: str
    merged_embeddings: str

class DataColumnsConfig(BaseModel):
    sequence_id: str
    feature_id: str
    target: str

class DataSplitConfig(BaseModel):
    test_size: float
    val_size: float

class DataConfig(BaseModel):
    paths: DataPathsConfig
    columns: DataColumnsConfig
    split: DataSplitConfig

class ViTConfig(BaseModel):
    patch_size: int
    projection_dim: int
    num_layers: int
    num_attention_heads: int
    mlp_units: List[int]
    dropout_rate: float

class CNNConfig(BaseModel):
    filters: int
    kernel_size: int
    pool_size: int
    activation: str

class FeatureBranchConfig(BaseModel):
    dense_units: List[int]
    activation: str

class FinalLayerConfig(BaseModel):
    dense_units: int
    activation: str

class RegularizationConfig(BaseModel):
    dropout_rate: float

class ModelConfig(BaseModel):
    vit: ViTConfig
    cnn: CNNConfig
    features: FeatureBranchConfig
    final: FinalLayerConfig
    regularization: RegularizationConfig

class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    min_lr: float

class FeatureSelectionConfig(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    device: str
    tree_method: str
    k: int

class FeatureNormalizationConfig(BaseModel):
    method: str

class FeatureProcessingConfig(BaseModel):
    selection: FeatureSelectionConfig
    normalization: FeatureNormalizationConfig

class ModelArtifactsOutputConfig(BaseModel):
    model_filename: str
    scaler_filename: str
    selector_filename: str

class OutputConfig(BaseModel):
    results_dir: str
    model_artifacts: ModelArtifactsOutputConfig

class SHAPConfig(BaseModel):
    enabled: bool
    num_background_samples: int
    num_test_samples: int
    max_display: int

class DeepDegConfig(BaseModel):
    reproducibility: ReproducibilityConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    features: FeatureProcessingConfig
    output: OutputConfig
    shap: SHAPConfig