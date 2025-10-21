#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lse_mt5_lora.py
---------------------
Fine-tuning local de mT5-small + LoRA para transformar texto ES -> LSE (glosa)
usando un bucle de entrenamiento propio (PyTorch) sin HuggingFace Trainer.

Dataset esperado (CSV):
    columnas: oracion_español, oracion_transformada_lse

Ejemplo de uso:
    python train_lse_mt5_lora.py \
        --csv /ruta/dataset.csv \
        --out ./mt5_lse_lora \
        --epochs 3 \
        --batch_size 8 \
        --grad_accum 2

Consejos:
- 25k pares + 3 épocas suele ir muy bien. Si el val_loss sigue bajando, prueba 4.
- Si te quedas sin VRAM: baja batch_size a 4, o MAX_*_LEN a 192.
"""

import os
os.environ["TRANSFORMERS_PREFER_SAFETENSORS"] = "1"  # fuerza safetensors
import math
import time
import json
import argparse
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Any

from contextlib import nullcontext

import torch
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------
# Utilidades
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert "oracion_español" in df.columns and "oracion_transformada_lse" in df.columns, \
        "El CSV debe tener columnas: oracion_español, oracion_transformada_lse"
    df["source"] = df["oracion_español"].astype(str).str.strip()
    df["target"] = df["oracion_transformada_lse"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")].dropna(subset=["source", "target"]).reset_index(drop=True)
    return df[["source", "target"]]

def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer, max_src_len=256, max_tgt_len=256):
    def add_prefix(batch):
        batch["input_text"] = ["es->lse: " + t for t in batch["source"]]
        batch["target_text"] = [t for t in batch["target"]]
        return batch

    raw = DatasetDict(
        train=Dataset.from_pandas(train_df),
        validation=Dataset.from_pandas(val_df)
    ).map(add_prefix, batched=True, remove_columns=[])

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_src_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_tgt_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=raw["train"].column_names)
    return tokenized

@dataclass
class TrainConfig:
    base_model: str = "google/mt5-small"
    out_dir: str = "./mt5_lse_lora"
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_src_len: int = 256
    max_tgt_len: int = 256
    eval_every_steps: int = 500
    patience: int = 3
    seed: int = 42
    fp16: bool = True  # auto-desactivamos si no hay CUDA
    num_workers: int = 2
    save_total_limit: int = 2

def build_dataloaders(tokenized, tokenizer, cfg: TrainConfig, device):
    # Collator SIN el modelo (para que sea pickeable en Windows)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    from torch.utils.data import DataLoader
    pin = (device.type == "cuda")

    # ⚠️ En Windows: num_workers=0 (spawn + pickling = dolor)
    effective_workers = 0 if os.name == "nt" else cfg.num_workers

    train_loader = DataLoader(
        tokenized["train"], batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collator, num_workers=effective_workers, pin_memory=pin, persistent_workers=False
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collator, num_workers=effective_workers, pin_memory=pin, persistent_workers=False
    )
    return train_loader, val_loader

def move_to_device(batch: Dict[str, Any], device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def has_valid_labels(batch: Dict[str, Any]) -> bool:
    # evitar batches con todas las labels == -100 (vacías)
    if "labels" not in batch:
        return False
    labels = batch["labels"]
    # si alguna fila tiene algún id != -100, lo aceptamos
    if labels.dim() == 1:
        return (labels != -100).any().item()
    return (labels != -100).any().item()

def save_lora(model: PeftModel, tokenizer, out_dir: str, step_or_epoch: str = None, keep_last: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    if step_or_epoch is None:
        save_dir = out_dir
    else:
        save_dir = os.path.join(out_dir, step_or_epoch)
        os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # limpieza: mantener solo los últimos 'keep_last' subdirectorios
    subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    # no tocar el root si es donde guardamos final
    dated = []
    for d in subdirs:
        p = os.path.join(out_dir, d)
        try:
            t = os.path.getmtime(p)
            dated.append((t, d))
        except Exception:
            pass
    dated.sort(reverse=True)
    for _, d in dated[keep_last:]:
        try:
            import shutil
            shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)
        except Exception:
            pass

def make_bad_words_ids(tok):
    # Bloquea artículos/preposiciones frecuentes en decodificación
    toks = [
        "▁el","▁la","▁los","▁las","▁un","▁una","▁unos","▁unas",
        "▁del","▁al",
        "▁de","▁a","▁en","▁con","▁por","▁para","▁sin","▁sobre","▁entre","▁hacia",
        "▁desde","▁según","▁contra","▁durante","▁mediante","▁tras","▁hasta"
    ]
    out = []
    for w in toks:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            out.append(ids)
    return out

def get_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    # Torch nuevo
    try:
        return torch.amp.autocast('cuda', enabled=True)
    except Exception:
        # Torch antiguo
        return torch.cuda.amp.autocast(enabled=True)

# ---------------------------
# Entrenamiento sin Trainer
# ---------------------------

def train_loop(cfg: TrainConfig, csv_path: str, val_csv_path: str = None):
    set_seed(cfg.seed)
    device = detect_device()
    print(f"[INFO] Device: {device}")

    # Tokenizer + datos
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)

    df = load_dataframe(csv_path)
    if val_csv_path:
        df_val = load_dataframe(val_csv_path)
        train_df = df
        val_df = df_val
    else:
        train_df, val_df = train_test_split(df, test_size=0.05, random_state=cfg.seed, shuffle=True)

    tokenized = build_datasets(train_df, val_df, tokenizer, cfg.max_src_len, cfg.max_tgt_len)

    # Modelo base + LoRA
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, use_safetensors=True)
    base.config.use_cache = False  # necesario con gradient checkpointing
    base.gradient_checkpointing_enable()
    model = get_peft_model(
        base,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q", "k", "v", "o", "wi", "wo"],  # mT5
            bias="none", task_type="SEQ_2_SEQ_LM"
        ),
    )
    model.to(device)
    print(f"[INFO] Trainable params (LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader = build_dataloaders(tokenized, tokenizer, cfg, device)

    # Optimizador + scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)

    total_steps = math.ceil(len(train_loader) / cfg.grad_accum) * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # AMP
    use_fp16 = cfg.fp16 and device.type == "cuda"
    # AMP con fallback a la API nueva si existe
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)  # fallback

    def _is_finite(t):
        return torch.isfinite(t).all().item() if torch.is_tensor(t) else math.isfinite(t)


    # Bad words para sanity de generación
    bad_words_ids = make_bad_words_ids(tokenizer)

    best_val = float("inf")
    steps_no_improve = 0
    global_step = 0

    # Logging a fichero (TSV)
    log_path = os.path.join(cfg.out_dir, "train_log.tsv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("step\tepoch\ttrain_loss\tval_loss\tlr\n")

    # media móvil exponencial para train
    ema = None
    def update_ema(x, alpha=0.1):
        nonlocal ema
        if x is None or not math.isfinite(x):
            return ema
        ema = x if ema is None else (alpha * x + (1 - alpha) * ema)
        return ema


    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    torch.backends.cudnn.benchmark = True

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            if not has_valid_labels(batch):
                # evita NaN por labels todas -100
                continue

            # --- Fwd + loss con AMP (si FP16) ---
            with get_autocast(use_fp16):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum

            # Si viene NaN/Inf, rehacer en FP32; si sigue mal, saltar batch
            if not _is_finite(loss):
                with nullcontext():
                    out = model(**batch)
                    loss = out.loss / cfg.grad_accum
                if not _is_finite(loss):
                    continue

            # Backward acumulado
            scaler.scale(loss).backward()
            running += float(loss.item()) if _is_finite(loss) else 0.0

            # Step cada grad_accum
            if step % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)

                # Descarta paso si hay gradientes no finitos
                grad_ok = True
                for p in model.parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all()):
                        grad_ok = False
                        break
                if not grad_ok:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Logging cada 50 pasos
                if global_step % 50 == 0:
                    avg = running / 50 if running > 0 and math.isfinite(running) else float('nan')
                    running = 0.0
                    ema_val = update_ema(avg)
                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix({"train_loss/EMA": f"{(ema_val if ema_val is not None else float('nan')):.4f}"},
                                    refresh=False)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{global_step}\t{epoch}\t{avg:.6f}\t\t{lr_now:.8f}\n")

                # Validación periódica (eval SIEMPRE en FP32)
                if global_step % cfg.eval_every_steps == 0:
                    val_loss = evaluate(model, val_loader, device, use_fp16=False)
                    print(f"\n[VAL] step {global_step} | val_loss={val_loss:.4f}")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{global_step}\t{epoch}\t\t{val_loss:.6f}\t{scheduler.get_last_lr()[0]:.8f}\n")

                    improved = val_loss < best_val - 1e-5
                    if improved:
                        best_val = val_loss
                        steps_no_improve = 0
                        save_lora(model, tokenizer, cfg.out_dir, step_or_epoch=f"step-{global_step}", keep_last=cfg.save_total_limit)
                    else:
                        steps_no_improve += 1
                        if steps_no_improve >= cfg.patience:
                            print("[INFO] Early stopping por paciencia.")
                            pbar.close()
                            finalize_and_save(model, tokenizer, cfg.out_dir)
                            quick_sanity_generate(model, tokenizer, device, bad_words_ids)
                            return
            pbar.update(1)

        pbar.close()

        # Validación al final de la época
        val_loss = evaluate(model, val_loader, device, use_fp16=False)
        print(f"[VAL] epoch {epoch} | val_loss={val_loss:.4f}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch_end\t{epoch}\t\t{val_loss:.6f}\t{scheduler.get_last_lr()[0]:.8f}\n")

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            steps_no_improve = 0
            save_lora(model, tokenizer, cfg.out_dir, step_or_epoch=f"epoch-{epoch}", keep_last=cfg.save_total_limit)
        else:
            steps_no_improve += 1
            if steps_no_improve >= cfg.patience:
                print("[INFO] Early stopping por paciencia (por épocas).")
                break

    finalize_and_save(model, tokenizer, cfg.out_dir)
    quick_sanity_generate(model, tokenizer, device, bad_words_ids)

def evaluate(model, val_loader, device, use_fp16: bool) -> float:
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = move_to_device(batch, device)
            if not has_valid_labels(batch):
                continue
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(**batch)
                loss = out.loss
            bs = batch["labels"].size(0)
            tot += loss.item() * bs
            n += bs
    return (tot / max(1, n))

def finalize_and_save(model, tokenizer, out_dir: str):
    print("[INFO] Guardando adapters finales…")
    save_lora(model, tokenizer, out_dir, step_or_epoch=None, keep_last=2)
    print(f"[OK] Adapters guardados en: {out_dir}")

def quick_sanity_generate(model, tokenizer, device, bad_words_ids):
    print("\n[Sanity] Generación de prueba:")
    prompts = [
        "¿Vas a venir mañana a mi casa?",
        "Luis es médico.",
        "No tengo dinero.",
        "El año pasado fui a Madrid.",
        "¿Qué compraste en la tienda?"
    ]
    gen = model.eval()
    for s in prompts:
        text = "es->lse: " + s.strip()
        inputs = tokenizer([text], return_tensors="pt").to(device)
        with torch.no_grad():
            out = gen.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,            # beam search
                bad_words_ids=bad_words_ids
                # (sin temperature, solo se usa en sampling)
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"ES : {s}\nLSE: {decoded}\n—")

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning mT5-small + LoRA para ES->LSE (glosa)")
    p.add_argument("--csv", required=True, help="Ruta a dataset CSV: oracion_español,oracion_transformada_lse")
    p.add_argument("--val_csv", default=None, help="Ruta a CSV de validación (opcional). Si no, 5%% split.")
    p.add_argument("--out", default="./mt5_lse_lora", help="Directorio de salida para adapters")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_src_len", type=int, default=256)
    p.add_argument("--max_tgt_len", type=int, default=256)
    p.add_argument("--eval_every_steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=3, help="Número de validaciones sin mejora antes de parar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_fp16", action="store_true", help="Desactiva FP16 aunque haya GPU")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig(
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        eval_every_steps=args.eval_every_steps,
        patience=args.patience,
        seed=args.seed,
        fp16=not args.no_fp16,
        num_workers=args.num_workers,
    )
    print("[CFG]", json.dumps(cfg.__dict__, ensure_ascii=False, indent=2))
    train_loop(cfg, csv_path=args.csv, val_csv_path=args.val_csv)

if __name__ == "__main__":
    main()
