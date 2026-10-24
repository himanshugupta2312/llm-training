import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def pack_texts(texts, tokenizer, block_size: int):
    token_stream = []
    for text in tqdm(texts, desc="Tokenizing", unit="rows"):
        text = " ".join(text.split())
        if not text:
            continue
        token_stream.extend(tokenizer.encode(text, add_special_tokens=False))
        token_stream.append(tokenizer.eos_token_id)

    blocks = []
    usable_tokens = len(token_stream) - (len(token_stream) % block_size)
    for start in range(0, usable_tokens, block_size):
        input_ids = token_stream[start : start + block_size]
        blocks.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )
    return blocks, usable_tokens


def load_wikitext(split: str, max_samples: int):
    return load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=f"{split}[:{max_samples}]",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare packed GPT-2 token blocks from WikiText-2."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum WikiText training rows to use (the full training split has about 36,000 rows).",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=1000,
        help="Maximum rows to take from WikiText's separate validation split.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokens per training example; 128 is a practical setting for a 4 GB GPU.",
    )
    args = parser.parse_args()
    if args.max_samples < 1 or args.validation_samples < 1 or args.max_length < 2:
        parser.error("sample counts must be positive and max-length must be at least 2")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading up to {args.max_samples:,} WikiText-2 training rows...")
    train_rows = load_wikitext("train", args.max_samples)
    print(f"Loading up to {args.validation_samples:,} separate validation rows...")
    validation_rows = load_wikitext("validation", args.validation_samples)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_examples, train_tokens = pack_texts(
        (row["text"] for row in train_rows), tokenizer, args.max_length
    )
    validation_examples, validation_tokens = pack_texts(
        (row["text"] for row in validation_rows), tokenizer, args.max_length
    )
    if not train_examples or not validation_examples:
        raise ValueError(
            "Not enough text to produce train and validation blocks; increase the sample counts."
        )

    torch.save(train_examples, PROCESSED_DIR / "train_tokens.pt")
    torch.save(validation_examples, PROCESSED_DIR / "validation_tokens.pt")

    metadata = {
        "dataset": "wikitext-2-raw-v1",
        "tokenizer": "gpt2",
        "max_length": args.max_length,
        "train_source_rows": len(train_rows),
        "validation_source_rows": len(validation_rows),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "validation_source": "official WikiText-2 validation split",
    }
    (PROCESSED_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    sample = train_examples[0]["input_ids"]
    print(f"Train blocks: {len(train_examples):,} ({train_tokens:,} tokens)")
    print(
        f"Validation blocks: {len(validation_examples):,} "
        f"({validation_tokens:,} tokens)"
    )
    print(f"Block shape: ({args.max_length},)")
    print("Sample decoded text:")
    decoded_sample = tokenizer.decode(sample[: min(100, len(sample))])
    print(decoded_sample.encode("ascii", errors="replace").decode("ascii"))
    print(f"Processed files saved under: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
