import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpt_from_scratch import GPTConfig, GPTLanguageModel, count_parameters


DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"


class TokenDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(example["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_batch(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=50256)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = padded_input_ids.clone()
    labels[padded_attention_mask == 0] = -100

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": labels,
    }


def load_data():
    train_path = DATA_DIR / "train_tokens.pt"
    val_path = DATA_DIR / "validation_tokens.pt"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed data not found. Run preprocess_data.py first.")

    train_data = torch.load(train_path, map_location="cpu")
    val_data = torch.load(val_path, map_location="cpu")
    return train_data, val_data


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(input_ids, labels)

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)

    model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(model, optimizer, epoch, step, loss, ckpt_path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )


def plot_losses(train_losses, val_losses, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train or resume a GPT-style language model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a training checkpoint.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Path for the best checkpoint (defaults to gpt_best.pt, or gpt_finetuned.pt when resuming).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_examples, val_examples = load_data()

    train_loader = DataLoader(
        TokenDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        TokenDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    config = GPTConfig(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        dropout=0.0,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        bias=True,
        use_gradient_checkpointing=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLanguageModel(config).to(device)
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_checkpoint_path = args.output_checkpoint
    if output_checkpoint_path is None:
        output_checkpoint_path = MODEL_DIR / ("gpt_finetuned.pt" if args.resume_from else "gpt_best.pt")

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    global_step = 0
    start_epoch = 0

    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint does not contain model_state_dict: {args.resume_from}")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
                param_group["weight_decay"] = args.weight_decay
        global_step = checkpoint.get("step", 0)
        start_epoch = checkpoint.get("epoch", -1) + 1

        best_val_loss = evaluate(model, val_loader, device)
        print(f"Resumed from {args.resume_from} - Validation loss: {best_val_loss:.4f}")
        save_checkpoint(
            model,
            optimizer,
            start_epoch - 1,
            global_step,
            best_val_loss,
            output_checkpoint_path,
        )

    output_stem = output_checkpoint_path.stem
    loss_curve_path = LOG_DIR / ("loss_curve.png" if output_stem == "gpt_best" else f"{output_stem}_loss_curve.png")
    history_path = LOG_DIR / (
        "training_history.json" if output_stem == "gpt_best" else f"{output_stem}_training_history.json"
    )

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(input_ids, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1
            epoch_steps += 1

            if global_step % args.checkpoint_every == 0 or global_step == 1:
                ckpt_path = MODEL_DIR / f"gpt_epoch{epoch+1}_step{global_step}.pt"
                save_checkpoint(model, optimizer, epoch, global_step, loss.item(), ckpt_path)

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        train_losses.append(avg_train_loss)

        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f} | Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, global_step, val_loss, output_checkpoint_path)

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    plot_losses(train_losses, val_losses, loss_curve_path)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(train_losses),
        "global_step": global_step,
    }
    history_path.write_text(json.dumps(history, indent=2))

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Loss curve saved to: {loss_curve_path}")
    print(f"Best checkpoint saved to: {output_checkpoint_path}")


if __name__ == "__main__":
    main()
