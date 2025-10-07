from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .postprocess import postprocess_gloss

def load_pipe(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return tok, mdl

def translate_list(texts, model_dir="outputs/mt5-lse", max_len=128):
    tok, mdl = load_pipe(model_dir)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    out = mdl.generate(**enc, max_length=max_len, num_beams=4)
    glosses = tok.batch_decode(out, skip_special_tokens=True)
    return [postprocess_gloss(g) for g in glosses]
