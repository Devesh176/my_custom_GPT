import os
# Must be set BEFORE torch is imported to take effect
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import glob
import shutil
import yaml
import torch
import matplotlib
matplotlib.use('Agg')   # headless-safe (Kaggle / no display)
import matplotlib.pyplot as plt
from pathlib import Path
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from gpt import GPT
from generate import generate_text
from tokenizer import Tokenizer
from dataloader import CustomDataset, dataloader_v1

_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Learning-rate scheduler: linear warm-up → cosine decay to 0
# ---------------------------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss / evaluation helpers
# ---------------------------------------------------------------------------
def calculate_loss(model, data_loader, device, num_batches):
    model.eval()
    total_loss = 0.0
    use_amp = device == 'cuda'
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda', enabled=use_amp):
                logits = model(inputs)
                loss   = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
            total_loss += loss.item()
    count = min(num_batches, len(data_loader))
    return total_loss / max(1, count)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss(model, train_loader, device, eval_iter)
        val_loss   = calculate_loss(model, val_loader,   device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_sample(model, tokenizer, device, start_context):
    model.eval()
    # torch.compile wraps the model; unwrap to access .positional_embedding if needed
    with torch.no_grad():
        text = generate_text(
            model=model, prompt=start_context, tokenizer=tokenizer,
            max_length=50, temperature=1.0, device=device
        )
        print(text.replace("\n", " "))
    model.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, optimizer, scheduler,
                num_epochs, eval_iter, tokenizer, start_context, eval_freq, config,
                checkpoint_dir=None, start_epoch=0):

    gradient_clip_val = config['GPT_CONFIG']['gradient_clip_val']
    grad_accum_steps  = config['TRAINING_CONFIG'].get('gradient_accumulation_steps', 1)
    log_interval      = config['TRAINING_CONFIG'].get('log_interval', 200)
    use_amp = (device == 'cuda')
    scaler  = GradScaler('cuda', enabled=use_amp)
    train_losses, val_losses = [], []
    running_loss = 0.0

    total_batches = len(train_loader)
    eff_batch = config['TRAINING_CONFIG']['batch_size'] * grad_accum_steps
    print(f"Starting training — {num_epochs} epoch(s), {total_batches} batches/epoch, "
          f"effective batch = {eff_batch}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=total_batches,
                    desc=f"Epoch {epoch+1}", dynamic_ncols=True, leave=True)

        for step, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast('cuda', enabled=use_amp):
                logits = model(inputs)
                loss   = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                ) / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % log_interval == 0:
                avg_loss     = running_loss / log_interval
                running_loss = 0.0
                current_lr   = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}", refresh=True)

        pbar.close()

        if (epoch + 1) % eval_freq == 0:
            train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter
            )
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nEpoch [{epoch+1}] | "
                  f"LR: {current_lr:.2e} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Save checkpoint — use the raw module if model was torch.compiled
        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

        print("Sample: ", end="")
        generate_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Dataset loader: WikiText-103
# ---------------------------------------------------------------------------
def load_wikitext103():
    from datasets import load_dataset
    print("Loading WikiText-103 …")
    ds = load_dataset("wikitext", "wikitext-103-v1")
    train_text = "\n".join(t for t in ds["train"]["text"]      if t.strip())
    val_text   = "\n".join(t for t in ds["validation"]["text"] if t.strip())
    print(f"  train: {len(train_text):,} chars | val: {len(val_text):,} chars")
    return train_text, val_text


# ---------------------------------------------------------------------------
# Kaggle persistence: resolve checkpoint directory across sessions
# ---------------------------------------------------------------------------
def resolve_checkpoint_dir(default_dir: Path) -> Path:
    """
    Returns the checkpoint directory to use, with Kaggle multi-session support.

    Priority:
      1. CHECKPOINT_DIR env var (set manually in notebook for full control)
      2. /kaggle/working/checkpoints  (if running on Kaggle — survives as output)
      3. default_dir (local / fallback)

    Also copies any checkpoints found in /kaggle/input/ into the working dir
    so training can resume from a previously saved session output.
    """
    # Allow full override via env var
    if 'CHECKPOINT_DIR' in os.environ:
        ckpt_dir = Path(os.environ['CHECKPOINT_DIR'])
    elif Path('/kaggle/working').exists():
        # Running on Kaggle — save outside the cloned repo so git clean doesn't wipe it
        ckpt_dir = Path('/kaggle/working/checkpoints')
    else:
        ckpt_dir = default_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Auto-import checkpoints from a previous Kaggle session saved as output/dataset
    # Files appear at /kaggle/input/<dataset-name>/checkpoints/*.pth
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists() and not sorted(ckpt_dir.glob('checkpoint_epoch_*.pth')):
        prev_ckpts = sorted(kaggle_input.glob('*/checkpoints/checkpoint_epoch_*.pth'))
        if prev_ckpts:
            print(f"Found {len(prev_ckpts)} checkpoint(s) in Kaggle input — copying to {ckpt_dir}")
            for f in prev_ckpts:
                shutil.copy(f, ckpt_dir)

    return ckpt_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config):
    torch.manual_seed(123)
    device = config['GPT_CONFIG']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to CPU")
        device = 'cpu'
        config['GPT_CONFIG']['device'] = 'cpu'

    # --- dataset ---
    train_text, val_text = load_wikitext103()

    # --- model ---
    model = GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # SPEED: torch.compile — fuses ops and generates optimised CUDA kernels.
    # First forward pass takes ~30-60s extra to compile; all subsequent passes
    # are ~20-35% faster. Requires PyTorch >= 2.0.
    if device == 'cuda' and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile() — first batch will be slow (~60s), rest faster…")
        model = torch.compile(model, mode='reduce-overhead')

    tokenizer     = Tokenizer("openai")
    start_context = "Every effort moves you"

    # --- dataloaders ---
    dl_cfg = config['data_load']
    tr_cfg = config['TRAINING_CONFIG']
    nw     = tr_cfg['num_workers']

    # SPEED: pin_memory → faster CPU→GPU transfer via DMA
    # SPEED: persistent_workers → workers stay alive between epochs (no fork overhead)
    # SPEED: prefetch_factor → workers pre-load batches while GPU is busy
    dl_kwargs = {}
    if device == 'cuda' and nw > 0:
        dl_kwargs = dict(pin_memory=True, persistent_workers=True, prefetch_factor=2)

    train_dataset = CustomDataset([train_text], tokenizer,
                                  block_size=dl_cfg['block_size'],
                                  stride=dl_cfg['stride'],
                                  max_length=dl_cfg['max_length'])
    val_dataset   = CustomDataset([val_text],   tokenizer,
                                  block_size=dl_cfg['block_size'],
                                  stride=dl_cfg['stride'],
                                  max_length=dl_cfg['max_length'])

    train_loader = dataloader_v1(train_dataset, batch_size=tr_cfg['batch_size'],
                                 shuffle=True,  num_workers=nw, **dl_kwargs)
    val_loader   = dataloader_v1(val_dataset,   batch_size=tr_cfg['batch_size'],
                                 shuffle=False, num_workers=nw, **dl_kwargs)
    print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # --- checkpoint dir (Kaggle-persistence-aware) ---
    checkpoint_dir = resolve_checkpoint_dir(_ROOT / 'checkpoints')
    start_epoch = 0
    existing = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if existing:
        latest = existing[-1]
        print(f"Resuming from checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        # Load into raw model (before compile wrapping)
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"  → Resuming from epoch {start_epoch + 1}")

    # --- optimiser & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg['learning_rate']),
        weight_decay=float(tr_cfg['weight_decay'])
    )
    total_steps  = len(train_loader) * tr_cfg['num_epochs']
    warmup_steps = config['GPT_CONFIG']['warmup_steps']
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if existing:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    print(f"Total steps: {total_steps:,} | Warmup: {warmup_steps:,} | "
          f"Start epoch: {start_epoch + 1} | Checkpoint dir: {checkpoint_dir}")

    # --- train ---
    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, optimizer=optimizer, scheduler=scheduler,
        num_epochs=tr_cfg['num_epochs'], eval_iter=tr_cfg['eval_iter'],
        tokenizer=tokenizer, start_context=start_context,
        eval_freq=tr_cfg['eval_freq'], config=config,
        checkpoint_dir=checkpoint_dir, start_epoch=start_epoch
    )

    return train_losses, val_losses, model, tokenizer


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(_ROOT / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_losses, val_losses, model, tokenizer = main(config)

    plot_loss(train_losses, val_losses,
              save_path=str(_ROOT / 'loss_plot.png'))

    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    ckpt = _ROOT / 'gpt_model.pth'
    torch.save(raw_model.state_dict(), ckpt)
    print(f"Model saved → {ckpt}")
