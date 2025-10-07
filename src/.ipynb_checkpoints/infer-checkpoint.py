# src/infer.py
import argparse
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.features import parse

def _noacc_lower(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower()

def _collect_source_names(text: str):
    doc = parse(text)
    names = []
    for t in doc:
        if t.pos_ == "PROPN":
            names.append((t.text, _noacc_lower(t.text)))
    return names

def _repair_names_with_hash(src_text: str, hyp_text: str) -> str:
    names = _collect_source_names(src_text)
    if not names:
        return hyp_text
    toks = hyp_text.split()
    for i, w in enumerate(toks):
        w_clean = w.lstrip("#")
        nw = _noacc_lower(w_clean)
        for orig, orig_na in names:
            # si coincide ignorando acentos o ya coincide en mayúsculas
            if nw == orig_na or w_clean.upper() == orig.upper():
                toks[i] = "#" + orig.upper()
                break
    return " ".join(toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    enc = tok(args.text, return_tensors="pt")
    out = model.generate(**enc, max_length=64)
    pred = tok.decode(out[0], skip_special_tokens=True)

    # --- post-proceso de nombres/#
    pred = _repair_names_with_hash(args.text, pred)

    print(pred)

if __name__ == "__main__":
    main()
