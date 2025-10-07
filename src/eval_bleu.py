# src/eval_bleu.py
import json, argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import re, sacrebleu
import unicodedata

def strip_accents(s): 
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def predict_batch(model, tok, texts, max_len=64):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=96)
    out = model.generate(**enc, max_length=max_len)
    return tok.batch_decode(out, skip_special_tokens=True)

def _norm_spaces(s): return re.sub(r"\s+", " ", s.strip())
def _strip_hash(s):  return s.replace("#", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_path", default="data/synthetic/test.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    test = load_jsonl(args.test_path)
    srcs = [x["src"] for x in test]
    refs = [x["tgt"] for x in test]

    hyps = predict_batch(model, tok, srcs)

    exact_strict = sum(_norm_spaces(h)==_norm_spaces(r) for h, r in zip(hyps, refs)) / len(refs)
    exact_nohash = sum(_norm_spaces(_strip_hash(h))==_norm_spaces(_strip_hash(r)) for h, r in zip(hyps, refs)) / len(refs)
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    print(f"Samples: {len(refs)}")
    print(f"Exact-match (estricto): {exact_strict*100:.2f}%")
    exact_nohash_noacc = sum(
        _norm_spaces(strip_accents(_strip_hash(h))) == _norm_spaces(strip_accents(_strip_hash(r)))
        for h, r in zip(hyps, refs)
    ) / len(refs)
    print(f"Exact-match (ignora '#', sin tildes): {exact_nohash_noacc*100:.2f}%")    
    print(f"BLEU: {bleu:.2f}")

    # Muestra 5 pares para inspección humana
    for i in range(5):
        print("\nSRC:", srcs[i])
        print("REF:", refs[i])
        print("HYP:", hyps[i])

if __name__ == "__main__":
    main()
