#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, torch, re, unicodedata
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE = "google/mt5-small"

# Mismo listado de palabras que no quieres que genere
def make_bad_words_ids(tok):
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

def postprocess_gloss(s: str) -> str:
    # Estandariza: mayúsculas + espacios simples; elimina dobles ?, !
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    s = s.upper()
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"!{2,}", "!", s)
    return s

def load_model(ckpt_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval().to(device)
    return tok, model

@torch.no_grad()
def generate_one(model, tok, text: str, device: torch.device, mode="beam") -> str:
    bad_ids = make_bad_words_ids(tok)
    prompt = "es->lse: " + text.strip()
    inputs = tok([prompt], return_tensors="pt").to(device)

    if mode == "beam":
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            bad_words_ids=bad_ids,
        )
    else:  # sampling
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            bad_words_ids=bad_ids,
        )

    pred = tok.decode(out[0], skip_special_tokens=True)
    return postprocess_gloss(pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./mt5_lse_lora", help="Ruta a los adapters LoRA")
    ap.add_argument("--mode", choices=["beam","sample"], default="beam")
    ap.add_argument("--text", help="Frase en español para convertir a LSE")
    ap.add_argument("--file", help="Ruta a fichero .txt con una frase por línea")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(args.ckpt, device)

    if args.text:
        print(generate_one(model, tok, args.text, device, mode=args.mode))
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                print("ES :", line)
                print("LSE:", generate_one(model, tok, line, device, mode=args.mode))
                print("—")
    else:
        print("Pasa --text 'oración...' o --file examples.txt")

if __name__ == "__main__":
    main()
