#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, unicodedata, math, torch, pandas as pd
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

try:
    import sacrebleu
except Exception:
    raise SystemExit("Instala sacrebleu: pip install sacrebleu")

BASE = "google/mt5-small"

ART_PREP = set(["EL","LA","LOS","LAS","UN","UNA","UNOS","UNAS","DEL","AL",
                "DE","A","EN","CON","POR","PARA","SIN","SOBRE","ENTRE","HACIA",
                "DESDE","SEGÚN","CONTRA","DURANTE","MEDIANTE","TRAS","HASTA"])

# Heurística muy simple para detectar formas verbales NO infinitivo
RE_CONJ = re.compile(r".*\b(É|ASTE|Ó|AMOS|ARON|Í|ISTE|IÓ|IMOS|IERON|ABA|ABAS|ÁBAMOS|ABAN|ÍA|ÍAS|ÍAMOS|ÍAN|ANDO|IENDO)\b", re.IGNORECASE)

def post(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper()

def make_bad_words_ids(tok):
    toks = [
        "▁el","▁la","▁los","▁las","▁un","▁una","▁unos","▁unas",
        "▁del","▁al","▁de","▁a","▁en","▁con","▁por","▁para","▁sin",
        "▁sobre","▁entre","▁hacia","▁desde","▁según","▁contra",
        "▁durante","▁mediante","▁tras","▁hasta"
    ]
    out = []
    for w in toks:
        ids = tok.encode(w, add_special_tokens=False)
        if ids:
            out.append(ids)
    return out

@torch.no_grad()
def predict_batch(model, tok, src_texts: List[str], device) -> List[str]:
    bad_ids = make_bad_words_ids(tok)
    prompts = ["es->lse: " + s.strip() for s in src_texts]
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
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
    dec = tok.batch_decode(out, skip_special_tokens=True)
    return [post(x) for x in dec]

def load_model(ckpt_dir: str, device):
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base, ckpt_dir).eval().to(device)
    return tok, model

def rule_checks(pred: str) -> Tuple[bool, bool]:
    # True si HAY infracción
    toks = pred.split()
    has_art_prep = any(t in ART_PREP for t in toks)
    has_conj      = bool(RE_CONJ.match(" " + pred + " "))
    return has_art_prep, has_conj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset con oracion_español,oracion_transformada_lse")
    ap.add_argument("--ckpt", default="./mt5_lse_lora", help="ruta adapters")
    ap.add_argument("--out_csv", default="predictions.csv")
    ap.add_argument("--limit", type=int, default=0, help="evalua sólo N filas (0=todas)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.rename(columns={"oracion_español":"source", "oracion_transformada_lse":"target"})
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"]!="") & (df["target"]!="")]
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_model(args.ckpt, device)

    B = 64
    preds = []
    for i in range(0, len(df), B):
        batch_src = df["source"].iloc[i:i+B].tolist()
        preds.extend(predict_batch(model, tok, batch_src, device))

    # post de referencias
    refs = [post(x) for x in df["target"].tolist()]

    # Métricas
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    # Chequeos de reglas
    bad_art, bad_conj = 0, 0
    for p in preds:
        a, c = rule_checks(p)
        bad_art += int(a)
        bad_conj += int(c)

    print(f"Size: {len(df)}")
    print(f"BLEU: {bleu.score:.2f}")
    print(f"chrF: {chrf.score:.2f}")
    print(f"Reglas - artículos/preps prohibidos: {bad_art/len(df)*100:.2f}%")
    print(f"Reglas - verbos NO infinitivo (heurístico): {bad_conj/len(df)*100:.2f}%")

    out = df.copy()
    out["prediction"] = preds
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"→ Guardado {args.out_csv}")

if __name__ == "__main__":
    main()
