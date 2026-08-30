import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import wandb
import pickle
import argparse

from pathlib import Path
from transformer import Transformer, TransformerConfig
from dataset import CustomDataset, collate_fn
from tokenizer import Tokenizer

current_file = Path(__file__).resolve()
BASE_DIR = current_file.parent.parent

SRC_PATH = BASE_DIR / "brown_cipher_tokenized.pkl"
TGT_PATH = BASE_DIR / "brown_plain_tokenized.pkl"
TOK_DIR = BASE_DIR / "tokenizer"
CIPHER_TOKENIZER = TOK_DIR / "brown_cipher2000_bits8.json"
PLAIN_TOKENIZER = TOK_DIR / "brown_plain5000_whitespace.json"

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    path
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, amp_enabled, amp_dtype):
    model.train()

    total_loss = 0.0

    for src, tgt in dataloader:
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(src.to(device), tgt_input.to(device))
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt_labels.to(device).reshape(-1)
            )


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                     # needed before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_loss(model, dataloader, criterion, device, amp_enabled, amp_dtype):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(src.to(device), tgt_input.to(device))
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_labels.to(device).reshape(-1)
                )

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    arch = "C1"
):
    model = model.to(device)

    # Ignore padding tokens when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="dumbkin31-anlp",
        # Set the wandb project where this run will be logged.
        project="anlp-a1",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": arch,
            "dataset": "brown",
            "epochs": num_epochs,
        },
    )

    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
            amp_dtype
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled,
            amp_dtype
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- Train Loss: {train_loss:.4f}; Val Loss: {val_loss:.4f}"
        )

        run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                train_loss,
                val_loss,
                "best_checkpoint.pt"
            )

    run.finish()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--rope", type=bool, required=True)
    parser.add_argument("--attention", type=str, required=True)
    parser.add_argument("--normalization", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()


    with open(SRC_PATH, "rb") as srcfile:
        src_data = pickle.load(srcfile)

    with open(TGT_PATH, "rb") as tgtfile:
        tgt_data = pickle.load(tgtfile)

    dataset = CustomDataset(src_data, tgt_data)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ci_tok = Tokenizer()
    ci_tok.load(CIPHER_TOKENIZER)

    pl_tok = Tokenizer()
    pl_tok.load(PLAIN_TOKENIZER)

    config = TransformerConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_len=350,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        rope=args.rope,
        attention=args.attention,
        normalization=args.normalization,
        dropout=0.1,
        src_vocab_size=len(ci_tok.encoder),
        tgt_vocab_size=len(pl_tok.encoder)
    )

    model = Transformer(config)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=1e-4,
        device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")