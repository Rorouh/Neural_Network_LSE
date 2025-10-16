
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_mt5_lora.py
----------------------------------
Ajuste fino (LoRA/PEFT) de mT5-small para la tarea ES->LSE.
Lee un CSV con columnas: id,source,target.

Uso rápido:
    pip install -U transformers datasets peft accelerate bitsandbytes sentencepiece pandas
    python finetune_mt5_lora.py --data dataset.csv --outdir outputs/mt5_lse_lora

Para generar después (inferencia):
    python finetune_mt5_lora.py --infer "Ayer estuvimos en el parque." --adapters outputs/mt5_lse_lora

Notas:
- Entrena solo los adapters LoRA (parámetros eficientes).
- Se añade un prefijo de instrucción: "es->lse: " para ayudar a mT5.
- Incluye un ejemplo de decodificación con palabras prohibidas (artículos/preps frecuentes).
"""
import os
import sys
import argparse
import random
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np

MODEL_NAME = os.getenv("BASE_MODEL", "google/mt5-small")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ARTICLES_PREPS = [
    "▁el","▁la","▁los","▁las","▁un","▁una","▁unos","▁unas",
    "▁del","▁al",
    "▁de","▁a","▁en","▁con","▁por","▁para","▁sin","▁sobre","▁entre","▁hacia",
    "▁desde","▁según","▁contra","▁durante","▁mediante","▁tras","▁hasta"
]

def make_bad_words_ids(tokenizer):
    bad = []
    for w in ARTICLES_PREPS:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            bad.append(ids)
    return bad

def split_train_val(df: pd.DataFrame, val_ratio=0.05):
    df = df.dropna(subset=["source","target"]).copy()
    df = df[df["target"].str.strip() != ""]
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(len(df)*val_ratio))
    val = df.iloc[:n_val].reset_index(drop=True)
    train = df.iloc[n_val:].reset_index(drop=True)
    return train, val

def build_datasets(csv_path: str):
    df = pd.read_csv(csv_path)
    train_df, val_df = split_train_val(df)

    # Prefijo de instrucción
    train_df["input_text"]  = "es->lse: " + train_df["source"].astype(str)
    train_df["target_text"] = train_df["target"].astype(str)

    val_df["input_text"]  = "es->lse: " + val_df["source"].astype(str)
    val_df["target_text"] = val_df["target"].astype(str)

    ds_train = Dataset.from_pandas(train_df[["input_text","target_text"]])
    ds_val   = Dataset.from_pandas(val_df[["input_text","target_text"]])
    return DatasetDict(train=ds_train, validation=ds_val)

def tokenize_function(examples, tokenizer, max_src_len=256, max_tgt_len=256):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_src_len, truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=max_tgt_len, truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    datasets = build_datasets(args.data)
    tokenized = datasets.map(
        lambda e: tokenize_function(e, tokenizer, args.max_src_len, args.max_tgt_len),
        batched=True, remove_columns=datasets["train"].column_names
    )

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q","v","k","o","wi","wo"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_cfg)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print("Adapters LoRA guardados en", args.outdir)

def infer(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, args.adapters)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    bad_words = make_bad_words_ids(tokenizer) if args.ban_words else None

    text = "es->lse: " + args.infer
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.2,
            num_beams=4,
            bad_words_ids=bad_words
        )
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV con columnas id,source,target")
    ap.add_argument("--outdir", default="outputs/mt5_lse_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-src-len", type=int, default=256)
    ap.add_argument("--max-tgt-len", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)

    # Inferencia
    ap.add_argument("--infer", type=str, help="Texto de prueba para inferencia")
    ap.add_argument("--adapters", type=str, help="Ruta a adapters LoRA guardados")
    ap.add_argument("--ban-words", action="store_true", help="Bloquear artículos/preposiciones frecuentes al generar")
    ap.add_argument("--max-new-tokens", type=int, default=128)

    args = ap.parse_args()

    if args.infer:
        if not args.adapters:
            print("Para inferencia necesitas --adapters con la carpeta de LoRA entrenada.", file=sys.stderr)
            sys.exit(2)
        infer(args)
    else:
        if not args.data:
            print("Falta --data para entrenar.", file=sys.stderr); sys.exit(2)
        os.makedirs(args.outdir, exist_ok=True)
        train(args)

if __name__ == "__main__":
    main()
