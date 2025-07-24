# DeepDeg: Hybrid Transformer-CNN Model for Predicting mRNA Degradation

## Introduction
DeepDeg is a novel deep learning framework designed for the accurate prediction of mRNA degradation rates. It utilizes a sophisticated multi-branch architecture that integrates Vision Transformer (ViT), Convolutional Neural Network (CNN), and Multi-Layer Perceptron (MLP) components to capture complex biological patterns.


## Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/turgaybulut/DeepDeg.git
    cd DeepDeg
    ```

2.  **Create and activate the conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate deepdeg
    ```

3.  **Install the project in editable mode:**
    ```bash
    pip install -e .
    ```
> **Note:** Please install and set up the PyBioMed package from [their repository](https://github.com/gadsbyfly/PyBioMed). It is only required for feature generation script.


## Usage

### Configuration

The main configuration file is `config.yaml`. You can modify this file to change the model, training, and data parameters.

### Scripts

Some scripts are independent and can be used with your own datasets, while others are specific to the DeepDeg project.

#### Dataset Scripts
The following scripts can be used with your own datasets by providing the necessary arguments.

*   **`scripts/feature_generator.py`**: Generate features from the data.
    *   **Usage**:
        ```bash
        python scripts/feature_generator.py <input_file> <output_file> [options]
        ```
    *   **Arguments**:
        *   `<input_file>`: Path to input CSV file with sequence data.
        *   `<output_file>`: Path to output CSV file for features.
    *   **Options**:
        *   `--features <type> [<type> ...]`: Feature types to extract (choices: `nucleic_acid`, `peptide`, `modlamp`, `enzymatic`, `all`). Default: `all`.
        *   `--batch-size <int>`: Batch size for processing. Default: `100`.
        *   `--num-cores <int>`: Number of CPU cores for parallel processing. Default: system's CPU count.
*   **`scripts/data_augmenter.py`**: Augment the data.
    *   **Usage**:
        ```bash
        python scripts/data_augmenter.py <command> <input_file> <output_file> [options]
        ```
    *   **Commands**:
        *   `sequences`: Augment sequence data.
        *   `features`: Augment feature data with Gaussian noise.
    *   **Arguments (common)**:
        *   `<input_file>`: Input CSV file.
        *   `<output_file>`: Output CSV file for augmented data.
    *   **Options (common)**:
        *   `--augmentation-factor <int>`: How many times to augment the dataset. Default: `2`.
        *   `--random-seed <int>`: Random seed for reproducibility. Default: `654`.
    *   **Options (for `sequences` command)**:
        *   `--utr-substitution-rate <float>`: Substitution rate for UTRs. Default: `0.015`.
        *   `--utr-insertion-rate <float>`: Insertion rate for UTRs. Default: `0.003`.
        *   `--utr-deletion-rate <float>`: Deletion rate for UTRs. Default: `0.002`.
        *   `--orf-substitution-rate <float>`: Synonymous codon substitution rate for ORFs. Default: `0.15`.
    *   **Options (for `features` command)**:
        *   `--noise-level <float>`: Noise level as a fraction of feature standard deviation. Default: `0.05`.
        *   `--id-columns <col> [<col> ...]`: Columns to exclude from noise application. Default: `ENSID HALFLIFE`.

#### Project Scripts
The following scripts are used for training and evaluating the DeepDeg model. They use the configuration from `config.yaml`.

*   **`scripts/train.py`**: Train the model.
*   **`scripts/evaluate.py`**: Evaluate the model.
*   **`scripts/train_fold.py`**: Run 10-fold cross-validation.

To run any script, use the following format:
```bash
python scripts/<script_name>.py
```

### Notebooks

The `notebooks/` directory contains Jupyter notebooks for generating embeddings using various pre-trained models:

*   **`mrnalm_embedding_generation.ipynb`**: Generate embeddings using the mRNA-LM model.
*   **`protbert_embedding_generation.ipynb`**: Generate embeddings using the ProtBERT model.
*   **`rnabert_embedding_generation.ipynb`**: Generate embeddings using the RNABERT model.

These notebooks can be run to pre-process data for the main DeepDeg pipeline or for independent research.


## Dataset

Our primary dataset can be downloaded from: https://drive.google.com/drive/folders/1YhHkKasumqMiGQv5D0Arexewy6LSxFAM