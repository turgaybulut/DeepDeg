# DeepDeg

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/turgaybulut/DeepDeg.git
   cd DeepDeg
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate deepdeg
   ```

3. **Install the project in editable mode:**
   ```bash
   pip install -e .
   ```

## Usage

### Training

To train the model, run the following command:

```bash
python scripts/train.py
```

### Evaluation

To evaluate the model, run the following command:

```bash
python scripts/evaluate.py
```

### Cross-Validation

To run cross-validation, use the following command:

```bash
python scripts/train_fold.py
```
