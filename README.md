# GPT Training Experiment

A personal learning project that implements a GPT-style causal language model in PyTorch, prepares WikiText-2 training examples, and includes a small training loop with validation and checkpointing. The repository also contains a separate Tkinter interface for running inference with a pretrained Qwen model.

This is an educational experiment, not a general-purpose LLM library or a production-ready model. The from-scratch model has not been pretrained at large scale.

## Repository contents

- `scripts/gpt_from_scratch.py` — GPT-style transformer model and a small forward/backward smoke demo.
- `scripts/preprocess_data.py` — downloads WikiText-2 and writes tokenized blocks under `data/processed/`.
- `scripts/train_gpt.py` — trains or resumes the model, evaluates validation loss, and writes checkpoints and plots.
- `scripts/inference_gui.py` — Tkinter chat interface using `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face.
- `scripts/evaluate_instruction_model.py` — runs a small repeatable prompt set against a pretrained causal language model.

## Setup

Use Python 3.10 or newer. Install the pinned dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Prepare data and train

Preprocess a small subset of WikiText-2:

```powershell
python scripts/preprocess_data.py --max-samples 1000 --validation-samples 200 --max-length 128
```

Run a short training smoke test:

```powershell
python scripts/train_gpt.py --epochs 1 --batch-size 2 --max-train-steps 10
```

The scripts create datasets, model checkpoints, and logs locally. These generated files are intentionally excluded from Git. Training a useful language model requires substantially more data and compute than this demo.

## Run the inference GUI

```powershell
python scripts/inference_gui.py
```

The first run downloads the configured pretrained model from Hugging Face. The GUI runs on CPU if CUDA is unavailable, but inference may be slow.

## Data and model notes

The preprocessing script uses WikiText-2 through Hugging Face Datasets and GPT-2 tokenization. Review the dataset and pretrained model licenses and terms before using them in other contexts. Do not commit downloaded datasets, local checkpoints, or private prompts.
