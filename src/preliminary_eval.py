"""
preliminary_eval.py

Preliminary evaluation for the ANLP Assignment 1 Transformer models.

Metrics required by the assignment:
    - Bit-Level Accuracy
    - Sequence Accuracy
    - Levenshtein Distance
    - BLEU
    - ROUGE

The assignment requires greedy decoding for evaluation.
BLEU and ROUGE are included for tokenized models only.

This script assumes:
    - src/test data are token-id sequences
    - target/test data are token-id sequences
    - 0 = PAD
    - 2 = BOS
    - 3 = EOS
    - your Transformer.forward(src, tgt_input) returns logits
    - your tokenizer can decode token IDs

This is intended for preliminary analysis. You can later move the
metric functions into utils.py as required by the assignment.
"""

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

# Standard metric libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

from transformer import Transformer, TransformerConfig
from dataset import CustomDataset, collate_fn
from tokenizer import Tokenizer


PAD = 0
BOS = 2
EOS = 3


# ---------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    """
    Greedy autoregressive decoding.

    Starts with BOS and repeatedly chooses the token with the highest
    logit until EOS is generated or max_len is reached.
    """
    model.eval()

    src = src.to(device)

    batch_size = src.size(0)

    tgt = torch.full(
        (batch_size, 1),
        BOS,
        dtype=torch.long,
        device=device
    )

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device
    )

    for _ in range(max_len - 1):

        output = model(src, tgt)

        # Last decoder position
        next_token = output[:, -1, :].argmax(dim=-1)

        # Once a sequence has produced EOS, keep it at EOS
        next_token = torch.where(
            finished,
            torch.full_like(next_token, EOS),
            next_token
        )

        tgt = torch.cat(
            [tgt, next_token.unsqueeze(1)],
            dim=1
        )

        finished |= next_token == EOS

        if finished.all():
            break

    return tgt


def remove_special_tokens(tokens):
    """
    Remove BOS/PAD and stop at EOS.
    """
    result = []

    for token in tokens:
        token = int(token)

        if token == EOS:
            break

        if token in (PAD, BOS):
            continue

        result.append(token)

    return result


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def bit_level_accuracy(pred, target):
    """
    Percentage of exact bit matches.

    For preliminary evaluation, predictions and targets are compared
    position-by-position after removing special tokens.

    Since this is a tokenized model, each decoded token is converted
    back to its string representation before comparing characters.
    """
    total_correct = 0
    total_bits = 0

    # This function is intended for decoded bit strings.
    for p, t in zip(pred, target):
        n = min(len(p), len(t))

        total_correct += sum(
            p[i] == t[i]
            for i in range(n)
        )

        total_bits += max(len(p), len(t))

    if total_bits == 0:
        return 0.0

    return total_correct / total_bits


def sequence_accuracy(predictions, targets):
    """
    Percentage of examples whose reconstructed sequence is exactly
    equal to the target sequence.
    """
    if len(targets) == 0:
        return 0.0

    correct = sum(
        pred == target
        for pred, target in zip(predictions, targets)
    )

    return correct / len(targets)


def levenshtein_distance(a, b):
    """
    Pure-Python Levenshtein edit distance.

    Kept here so the preliminary evaluator does not require another
    dependency.
    """
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(
                min(insertion, deletion, substitution)
            )

        previous = current

    return previous[-1]


def average_levenshtein(predictions, targets):
    if len(targets) == 0:
        return 0.0

    distances = [
        levenshtein_distance(pred, target)
        for pred, target in zip(predictions, targets)
    ]

    return sum(distances) / len(distances)


def bleu_score(predictions, targets):
    """
    Corpus-level BLEU using NLTK.

    Predictions/targets are lists of tokens.
    """
    if sentence_bleu is None:
        raise ImportError(
            "Install NLTK with: pip install nltk"
        )

    smoothie = SmoothingFunction().method1

    scores = []

    for pred, target in zip(predictions, targets):
        if len(pred) == 0:
            scores.append(0.0)
            continue

        score = sentence_bleu(
            [target],
            pred,
            smoothing_function=smoothie
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def rouge_scores(predictions, targets):
    """
    ROUGE-1, ROUGE-2 and ROUGE-L using rouge-score.

    Returns average F1 scores.
    """
    if rouge_scorer is None:
        raise ImportError(
            "Install rouge-score with: pip install rouge-score"
        )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False
    )

    totals = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }

    for pred, target in zip(predictions, targets):
        # rouge-score expects strings
        pred_text = " ".join(map(str, pred))
        target_text = " ".join(map(str, target))

        scores = scorer.score(
            target_text,
            pred_text
        )

        for name in totals:
            totals[name] += scores[name].fmeasure

    n = len(targets)

    if n == 0:
        return totals

    return {
        name: value / n
        for name, value in totals.items()
    }


# ---------------------------------------------------------------------
# Tokenizer / text reconstruction
# ---------------------------------------------------------------------

def ids_to_tokens(ids, tokenizer):
    """
    Convert token IDs into token strings and remove special tokens.
    """
    ids = remove_special_tokens(ids)

    return [
        tokenizer.decoder.get(int(token), "<unk>")
        for token in ids
    ]


def tokens_to_text(tokens):
    """
    Reconstruct text from BPE tokens.

    This follows the tokenizer format used in the assignment code:
    </w> marks the end of a word.
    """
    return "".join(tokens).replace("</w>", " ").strip()


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    max_decode_len=1000,
    max_examples=None
):
    model.eval()

    all_predictions = []
    all_targets = []

    num_seen = 0

    for src, tgt in dataloader:

        if max_examples is not None:
            remaining = max_examples - num_seen

            if remaining <= 0:
                break

            src = src[:remaining]
            tgt = tgt[:remaining]

        predicted = greedy_decode(
            model,
            src,
            max_decode_len,
            device
        )

        for pred_ids, target_ids in zip(predicted, tgt):

            pred_tokens = ids_to_tokens(
                pred_ids.tolist(),
                tokenizer
            )

            target_tokens = ids_to_tokens(
                target_ids.tolist(),
                tokenizer
            )

            all_predictions.append(pred_tokens)
            all_targets.append(target_tokens)

        num_seen += src.size(0)

    # Token-level representations
    seq_acc = sequence_accuracy(
        all_predictions,
        all_targets
    )

    lev = average_levenshtein(
        all_predictions,
        all_targets
    )

    bleu = bleu_score(
        all_predictions,
        all_targets
    )

    rouge = rouge_scores(
        all_predictions,
        all_targets
    )

    # Character-level reconstructed text.
    # Useful as a preliminary proxy for bit-level accuracy.
    pred_text = [
        tokens_to_text(x)
        for x in all_predictions
    ]

    target_text = [
        tokens_to_text(x)
        for x in all_targets
    ]

    char_acc = bit_level_accuracy(
        pred_text,
        target_text
    )

    return {
        "num_examples": len(all_targets),
        "bit_level_accuracy": char_acc,
        "sequence_accuracy": seq_acc,
        "average_levenshtein": lev,
        "bleu": bleu,
        "rouge1_f1": rouge["rouge1"],
        "rouge2_f1": rouge["rouge2"],
        "rougeL_f1": rouge["rougeL"],
        "predictions": all_predictions,
        "targets": all_targets,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--src",
        required=True,
        help="Path to source tokenized pickle"
    )

    parser.add_argument(
        "--tgt",
        required=True,
        help="Path to target tokenized pickle"
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to target/plaintext tokenizer JSON"
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=256
    )

    parser.add_argument(
        "--d_ff",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=6
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=352
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Use a small number for preliminary testing"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    with open(args.src, "rb") as f:
        src_data = pickle.load(f)

    with open(args.tgt, "rb") as f:
        tgt_data = pickle.load(f)

    dataset = CustomDataset(
        src_data,
        tgt_data
    )

    # IMPORTANT:
    # This preliminary script uses the LAST 10% as a test set.
    # For your final experiment, use the exact saved/random split
    # used during training so evaluation uses the same test examples.
    dataset = CustomDataset(src_data, tgt_data)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---------------------------------------------------------------
    # Load tokenizer
    # ---------------------------------------------------------------

    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer)

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Allows loading a checkpoint that is simply a state_dict
        state_dict = checkpoint

    # Infer vocabulary sizes from the checkpoint rather than hard-coding
    # them. This is important because your tokenizer vocabulary may not
    # be exactly 2000/5000 after special tokens are included.
    src_vocab_size = state_dict["encoder_embedding.weight"].shape[0]
    tgt_vocab_size = state_dict["decoder_embedding.weight"].shape[0]

    print("Checkpoint source vocab size:", src_vocab_size)
    print("Checkpoint target vocab size:", tgt_vocab_size)

    config = TransformerConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        rope=False,
        attention="mha",
        normalization="layernorm",
        dropout=0.0,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    )

    model = Transformer(config)

    # The Transformer already precomputes its sinusoidal positional
    # encoding, so positional_encoding.pe does not need to come from
    # the checkpoint.
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    print("Model loaded.")

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    results = evaluate(
        model,
        test_loader,
        tokenizer,
        device,
        max_decode_len=args.max_seq_len,
        max_examples=args.max_examples
    )

    print("\n===== PRELIMINARY RESULTS =====")
    print(f"Examples evaluated:      {results['num_examples']}")
    print(f"Bit-level accuracy:      {results['bit_level_accuracy']:.4f}")
    print(f"Sequence accuracy:       {results['sequence_accuracy']:.4f}")
    print(f"Avg. Levenshtein:        {results['average_levenshtein']:.4f}")
    print(f"BLEU:                    {results['bleu']:.4f}")
    print(f"ROUGE-1 F1:              {results['rouge1_f1']:.4f}")
    print(f"ROUGE-2 F1:              {results['rouge2_f1']:.4f}")
    print(f"ROUGE-L F1:              {results['rougeL_f1']:.4f}")

    # ---------------------------------------------------------------
    # Show examples
    # ---------------------------------------------------------------

    print("\n===== EXAMPLES =====")

    for i in range(min(10, len(results["predictions"]))):

        pred = tokens_to_text(
            results["predictions"][i]
        )

        target = tokens_to_text(
            results["targets"][i]
        )

        print(f"\nExample {i + 1}")
        print("TARGET:")
        print(target)
        print("PREDICTION:")
        print(pred)


if __name__ == "__main__":
    main()
