"""
Next-token prediction with a small Transformer built on basic (causal)
self-attention, written from scratch in PyTorch. Character-level.

The script downloads Tiny Shakespeare automatically (no curl needed), keeps a
small slice of it (300 train lines + 80 validation lines) and trains on it.

Run:
    python transformer_next_token.py
    python transformer_next_token.py --steps 2000 --prompt "The "
Options:
    --train_lines 300 --val_lines 80      size of the small dataset
    --train my_train.txt --val my_val.txt  use your own files instead
"""
import argparse
import math
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Hyperparameters ----------------
BLOCK_SIZE = 64    # context length
EMBED = 128        # embedding dimension
HEADS = 4
LAYERS = 4
DROPOUT = 0.1
BATCH = 32
LR = 3e-4


# ---------------- Data preparation ----------------
def prepare_small_dataset(n_train, n_val):
    """Download Tiny Shakespeare (once) and write train.txt / val.txt."""
    if not os.path.exists("input.txt"):
        print("Downloading Tiny Shakespeare ...")
        urllib.request.urlretrieve(DATA_URL, "input.txt")
    lines = [l for l in open("input.txt", encoding="utf-8").read().split("\n")
             if l.strip()][: n_train + n_val]
    with open("train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:n_train]) + "\n")
    with open("val.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines[n_train:]) + "\n")
    print(f"Created train.txt ({n_train} lines) and val.txt ({len(lines) - n_train} lines)")
    return "train.txt", "val.txt"


# ---------------- Model ----------------
class SelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with a causal mask."""
    def __init__(self, embed, heads, block_size, dropout):
        super().__init__()
        assert embed % heads == 0
        self.heads, self.hd = heads, embed // heads
        self.qkv = nn.Linear(embed, 3 * embed, bias=False)
        self.proj = nn.Linear(embed, embed)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, self.hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)   # (B, h, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, embed, heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed)
        self.attn = SelfAttention(embed, heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed)
        self.mlp = nn.Sequential(
            nn.Linear(embed, 4 * embed), nn.GELU(),
            nn.Linear(4 * embed, embed), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.mlp(self.ln2(x))    # residual + feed-forward
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, EMBED)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED)
        self.blocks = nn.Sequential(*[Block(EMBED, HEADS, BLOCK_SIZE, DROPOUT)
                                      for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_tokens, temperature=1.0):
        for _ in range(n_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


import time

# ---------------- Training ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="your own training text file")
    ap.add_argument("--val", default=None, help="your own validation text file")
    ap.add_argument("--train_lines", type=int, default=300)
    ap.add_argument("--val_lines", type=int, default=80)
    ap.add_argument("--full", action="store_true", help="use full Tiny Shakespeare dataset (90%% train, 10%% val)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--prompt", default="The ")
    args = ap.parse_args()

    if args.train and args.val:
        train_text = open(args.train, encoding="utf-8").read()
        val_text = open(args.val, encoding="utf-8").read()
    elif args.full:
        if not os.path.exists("input.txt"):
            print("Downloading Tiny Shakespeare ...", flush=True)
            urllib.request.urlretrieve(DATA_URL, "input.txt")
        full_text = open("input.txt", encoding="utf-8").read()
        n = int(0.9 * len(full_text))
        train_text, val_text = full_text[:n], full_text[n:]
        print(f"Using FULL Tiny Shakespeare dataset: {len(full_text):,} characters", flush=True)
    else:
        train_file, val_file = prepare_small_dataset(args.train_lines, args.val_lines)
        train_text = open(train_file, encoding="utf-8").read()
        val_text = open(val_file, encoding="utf-8").read()

    chars = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)

    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    val_data = torch.tensor(encode(val_text), dtype=torch.long)
    print(f"Train chars: {len(train_data):,} | Val chars: {len(val_data):,} | Vocab: {len(chars)}", flush=True)
    assert len(val_data) > BLOCK_SIZE + 1, "validation text too short"

    def get_batch(d):
        ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH,))
        x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])  # shifted by 1
        return x.to(device), y.to(device)

    model = MiniTransformer(len(chars)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    start_time = time.time()
    for step in range(args.steps + 1):
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                def eval_metrics(data):
                    losses, top1_accs, top5_accs = [], [], []
                    for _ in range(10):
                        x, y = get_batch(data)
                        logits, loss = model(x, y)
                        
                        # Top-1 accuracy
                        preds_top1 = logits.argmax(dim=-1)
                        top1 = (preds_top1 == y).float().mean().item() * 100
                        
                        # Top-5 accuracy
                        top5_preds = logits.topk(5, dim=-1).indices
                        top5 = (top5_preds == y.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100
                        
                        losses.append(loss.item())
                        top1_accs.append(top1)
                        top5_accs.append(top5)
                    return sum(losses) / len(losses), sum(top1_accs) / len(top1_accs), sum(top5_accs) / len(top5_accs)

                tl, ta1, ta5 = eval_metrics(train_data)
                vl, va1, va5 = eval_metrics(val_data)
            model.train()
            elapsed = time.time() - start_time
            rate = (step / elapsed) if elapsed > 0 else 0
            eta = ((args.steps - step) / rate) if rate > 0 else 0
            print(f"[{elapsed:5.1f}s] step {step:5d}/{args.steps} | "
                  f"train loss {tl:.3f} (top1: {ta1:.1f}%, top5: {ta5:.1f}%) | "
                  f"val loss {vl:.3f} (top1: {va1:.1f}%, top5: {va5:.1f}%)", flush=True)

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds ({total_time/60:.2f} minutes).", flush=True)

    model.eval()
    start = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    print("\n--- Generated text ---", flush=True)
    print(decode(model.generate(start, 300, temperature=0.8)[0].tolist()), flush=True)


if __name__ == "__main__":
    main()
