# Transformer-Based De Novo Drug Molecule Generation

This project uses a Transformer encoder-decoder model to generate molecular analogues from input SMILES strings. The model is trained on canonicalized SMILES data and uses self-attention to learn relationships between atoms/tokens.

## Project Files

- `Transformer_architecture - Final.ipynb` — main Google Colab notebook containing the Transformer implementation, training, and molecule generation.
- `Using Transformers for De Novo Drug Molecule Generation(1).pdf` — project report and methodology.

## Requirements

The notebook was developed for **Google Colab** and uses a **GPU** for training.

Main Python packages:

- Python 3
- PyTorch
- NumPy
- RDKit
- Matplotlib/standard Python libraries used by Colab

The notebook installs RDKit with:

```bash
!pip install rdkit-pypi
```

A **T4 GPU** was used for the reported implementation.

## Dataset

The notebook expects a CSV file named:

```text
Compounds2.csv
```

with the SMILES strings in the **first column**.

The notebook currently reads the file from:

```text
/content/drive/MyDrive/Compounds2.csv
```

Therefore, upload `Compounds2.csv` to your Google Drive, preferably directly inside `My Drive`.

The preprocessing in the notebook:

1. Reads the first column of the CSV.
2. Keeps SMILES strings up to the configured maximum length.
3. Uses an 85:15 training/validation split.
4. Builds a character-level vocabulary from the training data.
5. Adds `<START>`, `<END>`, and `<PADDING>` tokens.

> **Note:** The report describes a dataset of approximately 3 million molecules, but the current notebook explicitly limits the loaded data to the first **100,000** entries before training. This is the version that will actually run from the supplied notebook.

## Running the Notebook

### 1. Open Google Colab

Upload/open:

```text
Transformer_architecture - Final.ipynb
```

in Google Colab.

### 2. Enable GPU

In Colab:

```text
Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

A GPU is strongly recommended because the model uses a 512-dimensional embedding, 8 attention heads, 3 Transformer layers, and a feed-forward dimension of 2048.

### 3. Mount Google Drive

Run the notebook cells in order.

The notebook mounts Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Make sure the dataset exists at:

```text
/content/drive/MyDrive/Compounds2.csv
```

If your file is stored elsewhere, change the path in the dataset-loading cell.

### 4. Install RDKit

Run:

```bash
!pip install rdkit-pypi
```

The notebook also contains a cell that clones:

```text
https://github.com/rajkumar1501/drug_analog_data
```

This is included in the original notebook, but the main training data is loaded from `Compounds2.csv`.

### 5. Run the Model

Run the notebook cells sequentially from the beginning.

The main model configuration is:

```text
Embedding dimension : 512
Attention heads     : 8
Transformer layers  : 3
FFN dimension       : 2048
Dropout             : 0.2
Batch size          : 30
Epochs              : 5
Learning rate       : 1e-4
Maximum sequence    : 102 characters
```

The model is trained using cross-entropy loss with the padding tokens ignored.

The reported training run achieved a cross-entropy loss of approximately **0.09** after 5 epochs.

## Generating a Molecule

After training, the notebook defines:

```python
produce_new(new_mol)
```

You can enter a SMILES string when prompted:

```text
Enter a chemical SMILES(length < 100):
```

For example:

```text
CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C
```

The Transformer then generates a new SMILES sequence autoregressively.

The notebook converts both the input and generated SMILES into RDKit molecules and displays their structures.

### Important

The notebook warns that performance may degrade for SMILES strings longer than approximately **80 characters**, even though the configured maximum sequence length is 102.

The generated SMILES should be checked with RDKit before treating them as valid molecules.

## Model Architecture

The implementation contains:

- Token/character embeddings
- Positional encoding
- Multi-head self-attention
- Encoder-decoder cross-attention
- Feed-forward layers with ReLU
- Residual connections
- Layer normalization
- Dropout
- Linear output projection

The decoder generates the output one token at a time using an autoregressive procedure.

## Evaluation

The project evaluates generated molecules using:

- **Validity**
- **Uniqueness**
- **Novelty**

The report compares the Transformer with LSTM and GAN models using the MOSES platform.

Reported results for the Transformer:

| Metric | Transformer |
|---|---:|
| Validity | 97.55% |
| Uniqueness @1k | 96.25% |
| Uniqueness @10k | 98.85% |
| Novelty | 98.45% |

The Transformer achieved higher validity and novelty than the reported LSTM and GAN baselines.

## Troubleshooting

### `FileNotFoundError: Compounds2.csv`

Check that the file exists at:

```text
/content/drive/MyDrive/Compounds2.csv
```

or update `csv_file_path` in the notebook.

### CUDA/GPU errors

Make sure a GPU is enabled in:

```text
Runtime → Change runtime type
```

You can also check:

```python
torch.cuda.is_available()
```

It should return:

```text
True
```

### RDKit import error

Run:

```bash
!pip install rdkit-pypi
```

and restart the Colab runtime if necessary.

### Out-of-memory errors

The model is relatively large. Reduce:

```python
batch_size = 30
```

to a smaller value such as:

```python
batch_size = 10
```

if the available GPU memory is insufficient.

## Reference

Project report:

**Using Transformers for De Novo Drug Molecule Generation**

The original implementation was developed and run in Google Colab using a T4 GPU.
